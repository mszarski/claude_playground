"""Multimodal (text + image) conditioning extension for GenCAD.

GenCAD's pipeline is:  image --CCIP--> image latent --diffusion prior--> CSR latent --CSR decoder--> CAD sequence.
This package replaces the single-condition prior with a multi-condition prior that accepts any subset of
{text, image, GenCAD-CCIP image latent} and supports per-modality classifier-free guidance.

Modules
-------
- cond_encoders: frozen pretrained encoders (SigLIP2 / CLIP / any HF text encoder) that turn raw text or
  images into fixed embeddings, cached once to disk.
- fusion: ConditionFuser, which maps a variable set of modality embeddings to one conditioning vector,
  with learned null tokens and modality dropout (needed for classifier-free guidance).
- denoiser: AdaLN-modulated residual MLP denoiser over the 256-d CSR latent.
- diffusion: self-contained Gaussian diffusion (cosine schedule, x0-prediction, DDPM/DDIM sampling,
  compositional guidance over modalities).
- data: dataset of cached embeddings keyed by DeepCAD uid.
- gencad_io: loading GenCAD checkpoints, decoding latents to CAD vectors and exporting STEP/STL.
"""

MODALITIES = ("image", "text", "ccip_image")
