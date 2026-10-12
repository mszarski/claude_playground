"""Multi-condition latent diffusion prior with compositional classifier-free guidance.

Target x0 is GenCAD's CSR latent (256-d, inside [-1, 1] because the CSR bottleneck ends in tanh).
The condition is any subset of modalities, fused by ConditionFuser.

Guidance (sampling):
  chain       (InstructPix2Pix-style, order matters):
      x0 = f(∅) + w1 [f(m1) - f(∅)] + w2 [f(m1,m2) - f(m1)] + ...
  independent (composable diffusion):
      x0 = f(∅) + Σ_k wk [f(mk) - f(∅)]
All required condition sets are evaluated in one batched forward pass.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

from .denoiser import LatentDenoiser
from .fusion import ConditionFuser


def cosine_alphas_cumprod(T, s=0.008):
    steps = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
    ac = f / f[0]
    betas = (1 - ac[1:] / ac[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, dim=0).float()


class MultiCondDiffusionPrior(nn.Module):
    def __init__(self, modality_dims: dict, z_dim=256, timesteps=1000, objective="pred_x0",
                 fuser_kwargs=None, denoiser_kwargs=None, contrastive_weight=0.0, clip_range=1.0):
        super().__init__()
        assert objective in ("pred_x0", "pred_eps", "pred_v")
        fuser_kwargs = dict(fuser_kwargs or {})
        denoiser_kwargs = dict(denoiser_kwargs or {})
        self.fuser = ConditionFuser(modality_dims, **fuser_kwargs)
        self.denoiser = LatentDenoiser(z_dim=z_dim, d_cond=self.fuser.d_model, **denoiser_kwargs)
        self.z_dim, self.T, self.objective = z_dim, timesteps, objective
        self.clip_range = clip_range
        self.register_buffer("ac", cosine_alphas_cumprod(timesteps))
        self.contrastive_weight = contrastive_weight
        if contrastive_weight > 0:
            # auxiliary CLIP-style alignment between fused condition and target CAD latent
            self.c_head = nn.Linear(self.fuser.d_model, 256)
            self.z_head = nn.Sequential(nn.Linear(z_dim, 256), nn.SiLU(), nn.Linear(256, 256))
            self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.config = dict(modality_dims=modality_dims, z_dim=z_dim, timesteps=timesteps, objective=objective,
                           fuser_kwargs=fuser_kwargs, denoiser_kwargs=denoiser_kwargs,
                           contrastive_weight=contrastive_weight, clip_range=clip_range)

    @property
    def modalities(self):
        return self.fuser.names

    # ---------------------------------------------------------------- helpers
    def _coef(self, t):
        a = self.ac[t][:, None]
        return a.sqrt(), (1 - a).sqrt()

    def _to_x0(self, out, x_t, t):
        sa, s1a = self._coef(t)
        if self.objective == "pred_x0":
            return out
        if self.objective == "pred_eps":
            return (x_t - s1a * out) / sa
        return sa * x_t - s1a * out  # pred_v

    # ---------------------------------------------------------------- training
    def forward(self, z0, embeds: dict, present: dict | None = None):
        """Returns (loss, logs). present: {name: (B,) bool} marks which samples actually have a modality."""
        B, device = z0.shape[0], z0.device
        if present is None:
            present = {m: torch.ones(B, dtype=torch.bool, device=device) for m in embeds}
        keep = self.fuser.sample_keep_mask(present, B, device) if self.training else present
        cond = self.fuser(embeds, keep, batch=B)

        t = torch.randint(0, self.T, (B,), device=device)
        noise = torch.randn_like(z0)
        sa, s1a = self._coef(t)
        x_t = sa * z0 + s1a * noise
        out = self.denoiser(x_t, t, cond)
        target = {"pred_x0": z0, "pred_eps": noise, "pred_v": sa * noise - s1a * z0}[self.objective]
        loss = F.mse_loss(out, target)
        logs = {"diff_loss": loss.detach()}

        if self.contrastive_weight > 0:
            has_any = torch.stack(list(keep.values()), 0).any(0)
            if has_any.sum() > 1:
                c = F.normalize(self.c_head(cond[has_any]), dim=-1)
                z = F.normalize(self.z_head(z0[has_any]), dim=-1)
                logits = self.logit_scale.exp().clamp(max=100) * c @ z.t()
                labels = torch.arange(len(c), device=device)
                closs = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
                loss = loss + self.contrastive_weight * closs
                logs["clip_loss"] = closs.detach()
        logs["loss"] = loss.detach()
        return loss, logs

    # ---------------------------------------------------------------- sampling
    def _cond_sets(self, order, mode):
        if mode == "chain":
            return [tuple(order[:k]) for k in range(len(order) + 1)]
        if mode == "independent":
            return [()] + [(m,) for m in order]
        raise ValueError(mode)

    def _guided_x0(self, x, t, cond_bank, order, weights, mode):
        n_sets = len(cond_bank)
        B = x.shape[0]
        xs = x.repeat(n_sets, 1)
        ts = t.repeat(n_sets)
        outs = self.denoiser(xs, ts, torch.cat(cond_bank, 0))
        x0s = self._to_x0(outs, xs, ts).view(n_sets, B, -1)
        res = x0s[0].clone()
        for k, m in enumerate(order, start=1):
            base = x0s[k - 1] if mode == "chain" else x0s[0]
            res = res + weights.get(m, 1.0) * (x0s[k] - base)
        return res

    @torch.no_grad()
    def sample(self, embeds: dict, guidance: dict | None = None, order=None, mode="chain", steps=100,
               eta=0.0, batch=None, generator=None, return_trajectory=False):
        """
        embeds:   {name: (B, d)} conditions to use (any subset of self.modalities).
        guidance: {name: weight}; 1.0 == plain conditional sampling, >1 amplifies that modality.
        order:    order of modalities for chain guidance (default: as in self.modalities).
        steps:    number of DDIM steps (<= T). eta=0 deterministic DDIM, eta=1 ~ DDPM.
        """
        guidance = guidance or {}
        order = [m for m in (order or self.modalities) if m in embeds]
        B = batch or next(iter(embeds.values())).shape[0]
        device = self.ac.device

        cond_bank = []
        for s in self._cond_sets(order, mode):
            keep = {m: torch.full((B,), m in s, dtype=torch.bool, device=device) for m in self.modalities}
            cond_bank.append(self.fuser(embeds, keep, batch=B))

        x = torch.randn(B, self.z_dim, device=device, generator=generator)
        ts = torch.linspace(self.T - 1, 0, steps, device=device).round().long()
        traj = []
        for i, t_cur in enumerate(ts):
            t = t_cur.expand(B)
            x0 = self._guided_x0(x, t, cond_bank, order, guidance, mode)
            if self.clip_range:
                x0 = x0.clamp(-self.clip_range, self.clip_range)
            a_t = self.ac[t_cur]
            eps = (x - a_t.sqrt() * x0) / (1 - a_t).sqrt()
            if i == len(ts) - 1:
                x = x0
                break
            a_prev = self.ac[ts[i + 1]]
            sigma = eta * ((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev)).sqrt()
            dir_xt = (1 - a_prev - sigma ** 2).clamp(min=0).sqrt() * eps
            noise = torch.randn(x.shape, device=device, generator=generator) if eta > 0 else 0.0
            x = a_prev.sqrt() * x0 + dir_xt + sigma * noise
            if return_trajectory:
                traj.append(x0)
        return (x, traj) if return_trajectory else x

    @torch.no_grad()
    def score(self, embeds: dict, z):
        """Cosine similarity (B,) between the given conditions and CAD latents z (B, z_dim), from the contrastive
        heads trained alongside the denoiser. Used to rerank samples (mm.rerank.best_of_n)."""
        assert self.contrastive_weight > 0, "this prior was trained without contrastive heads"
        B = z.shape[0]
        keep = {m: torch.full((B,), m in embeds, dtype=torch.bool, device=z.device) for m in self.modalities}
        c = F.normalize(self.c_head(self.fuser(embeds, keep, batch=B)), dim=-1)
        return (c * F.normalize(self.z_head(z), dim=-1)).sum(-1)

    # ---------------------------------------------------------------- io
    def save(self, path, **extra):
        torch.save({"config": self.config, "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["config"])
        sd = ckpt.get("ema_state_dict") or ckpt["state_dict"]
        model.load_state_dict(sd)
        return model, ckpt
