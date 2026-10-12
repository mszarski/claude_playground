"""Best-of-N sampling: draw N latents per condition, decode each to a CAD program, re-encode the decoded program
with the CSR encoder, and keep the candidate the prior's contrastive head scores closest to the condition.

Scoring the *re-encoded decoded program* rather than the sampled latent matters: the CSR decoder picks each
coordinate token independently, so a latent that sits between two plausible shapes can decode to a mix of both
(e.g. a rectangle whose corners come from two different widths, i.e. a trapezoid). Re-encoding exposes that.
"""
import numpy as np
import torch

from .gencad_io import decode_latents, encode_cad


def _pad(vec, max_len, eos_vec):
    return np.concatenate([vec, eos_vec[None].repeat(max_len - len(vec), 0)], 0)


@torch.no_grad()
def best_of_n(prior, csr, embeds: dict, n=8, **sample_kw):
    """embeds: {name: (B, d)}. Returns (vecs, z, scores): the chosen decoded program per input (None if every
    candidate failed to decode), its sampled latent (B, z_dim), and its score (B,), -inf when nothing decoded."""
    from cadlib.macro import EOS_VEC, MAX_TOTAL_LEN
    B = next(iter(embeds.values())).shape[0]
    rep = {m: e.repeat_interleave(n, 0) for m, e in embeds.items()}
    z = prior.sample(rep, **sample_kw)                                   # (B*n, z_dim), candidates grouped per input
    vecs = decode_latents(csr, z)
    ok = [i for i, v in enumerate(vecs) if v is not None]
    scores = torch.full((B * n,), float("-inf"), device=z.device)
    if ok:
        padded = np.stack([_pad(vecs[i], MAX_TOTAL_LEN, EOS_VEC) for i in ok])
        cmd = torch.tensor(padded[:, :, 0], dtype=torch.long, device=z.device)
        args = torch.tensor(padded[:, :, 1:], dtype=torch.long, device=z.device)
        z_re = torch.cat([encode_cad(csr, cmd[s: s + 512], args[s: s + 512]) for s in range(0, len(ok), 512)])
        idx = torch.tensor(ok, device=z.device)
        scores[idx] = prior.score({m: e[idx] for m, e in rep.items()}, z_re.float())
    best = scores.view(B, n).argmax(1) + torch.arange(B, device=z.device) * n
    return [vecs[i] for i in best.tolist()], z[best], scores[best]
