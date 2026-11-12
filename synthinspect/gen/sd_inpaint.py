"""Shared Stable Diffusion inpainting machinery for the DefectFill and MAGIC re-implementations.

Both methods fine-tune the same inpainting backbone with LoRA and generate with
the same sampler, so they differ only where the papers differ:

DefectFill (CVPR 2025)
    LoRA rank 8 on UNet attention (lr 2e-4) and text encoder (lr 4e-5).
    Loss = 0.5 * L_def + 0.2 * L_obj + 0.05 * L_attn
      L_def : noise loss inside the defect mask, prompt "A photo of sks"
      L_obj : noise loss over random boxes plus the defect, prompt "A <object> with sks",
              background weighted by alpha < 1
      L_attn: cross-attention map of "sks" (decoder layers, averaged) vs the defect mask
    Generation: fixed guidance, several candidates per mask, keep the one with the
    highest LPIPS to the clean part inside the mask (Low-Fidelity Selection).

MAGIC (CVPR 2026 Findings), adapted to LoRA because full DreamBooth does not fit 8 GB
    Plain inpainting noise loss on the whole image, prompt "sks", text encoder frozen.
    Gaussian Prompt Perturbation: prompt embedding + N(0, sigma^2), sigma = 1.0,
    at training AND generation.
    Spatially Adaptive Guidance: scale 7.5 outside the mask; inside, a cosine ramp
    from w_m = 4 at the first (noisiest) step up to 7.5 at the last. The paper's
    text and its Eq. 6 disagree on direction; this follows the text.

The components are passed in, so tests can run the full training and sampling
code on a tiny randomly initialised UNet on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PLACEHOLDER = "sks"


# --------------------------------------------------------------------------- components


@dataclass
class SDComponents:
    unet: torch.nn.Module
    vae: torch.nn.Module
    text_encoder: torch.nn.Module
    tokenizer: Callable
    train_scheduler: object  # DDPMScheduler
    sample_scheduler: object  # DDIMScheduler
    device: torch.device
    dtype: torch.dtype
    resolution: int = 512

    @property
    def vae_scale(self) -> float:
        return float(getattr(self.vae.config, "scaling_factor", 0.18215))

    @property
    def latent_factor(self) -> int:
        return 2 ** (len(self.vae.config.block_out_channels) - 1)


def load_components(base_model: str, device: str | None = None, resolution: int = 512) -> SDComponents:
    """Load an inpainting checkpoint (9-channel UNet) from the Hugging Face Hub or a local folder."""
    from diffusers import DDIMScheduler, DDPMScheduler, StableDiffusionInpaintPipeline

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.float16 if dev.type == "cuda" else torch.float32
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        base_model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False, use_safetensors=True
    )
    if pipe.unet.config.in_channels != 9:
        raise ValueError(f"{base_model} is not an inpainting model (UNet has {pipe.unet.config.in_channels} input channels)")
    comp = SDComponents(
        unet=pipe.unet.to(dev),
        vae=pipe.vae.to(dev),
        text_encoder=pipe.text_encoder.to(dev),
        tokenizer=pipe.tokenizer,
        train_scheduler=DDPMScheduler.from_config(pipe.scheduler.config),
        sample_scheduler=DDIMScheduler.from_config(pipe.scheduler.config),
        device=dev,
        dtype=dtype,
        resolution=resolution,
    )
    for m in (comp.unet, comp.vae, comp.text_encoder):
        m.requires_grad_(False)
    return comp


def token_ids(comp: SDComponents, prompts: list[str]) -> torch.Tensor:
    tok = comp.tokenizer(prompts, padding="max_length", max_length=comp.tokenizer.model_max_length, truncation=True, return_tensors="pt")
    return tok.input_ids.to(comp.device)


def placeholder_positions(comp: SDComponents, prompt: str) -> list[int]:
    ids = token_ids(comp, [prompt])[0].tolist()
    target = comp.tokenizer(PLACEHOLDER, add_special_tokens=False).input_ids
    for i in range(len(ids) - len(target) + 1):
        if ids[i : i + len(target)] == list(target):
            return list(range(i, i + len(target)))
    raise ValueError(f"Placeholder {PLACEHOLDER!r} not found in prompt {prompt!r}")


def encode_text(comp: SDComponents, prompts: list[str]) -> torch.Tensor:
    return comp.text_encoder(token_ids(comp, prompts))[0]


# --------------------------------------------------------------------------- attention capture


@dataclass
class AttnStore:
    enabled: bool = False
    token_idx: list[int] = field(default_factory=list)
    maps: list[torch.Tensor] = field(default_factory=list)

    def reset(self, token_idx: list[int] | None = None, enabled: bool = True) -> None:
        self.maps = []
        self.enabled = enabled
        if token_idx is not None:
            self.token_idx = token_idx


class CaptureCrossAttnProcessor:
    """Diffusers' classic AttnProcessor, plus recording of the placeholder token's attention column."""

    def __init__(self, store: AttnStore):
        self.store = store

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, *args, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)
        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        is_cross = encoder_hidden_states is not None
        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        probs = attn.get_attention_scores(query, key, attention_mask)
        if self.store.enabled and is_cross and self.store.token_idx:
            col = probs[..., self.store.token_idx].sum(-1)  # (batch*heads, hw)
            self.store.maps.append(col.view(batch_size, attn.heads, -1).mean(1))  # (batch, hw)
        hidden_states = torch.bmm(probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def install_attention_capture(unet: torch.nn.Module) -> AttnStore:
    """Record the placeholder's cross-attention in the decoder (up_blocks), as DefectFill's L_attn needs."""
    store = AttnStore()
    procs = {}
    for name, proc in unet.attn_processors.items():
        procs[name] = CaptureCrossAttnProcessor(store) if (name.startswith("up_blocks") and "attn2" in name) else proc
    unet.set_attn_processor(procs)
    return store


def attention_map(store: AttnStore, size: int) -> torch.Tensor:
    """Average the recorded maps at one resolution and scale each to [0, 1]. Returns (batch, size, size)."""
    if not store.maps:
        raise RuntimeError("No attention maps were recorded")
    ups = []
    for m in store.maps:
        side = int(math.isqrt(m.shape[1]))
        ups.append(F.interpolate(m.view(-1, 1, side, side).float(), size=(size, size), mode="bilinear", align_corners=False))
    a = torch.stack(ups).mean(0)[:, 0]
    return a / a.flatten(1).amax(1).clamp_min(1e-8).view(-1, 1, 1)


# --------------------------------------------------------------------------- LoRA


UNET_TARGETS = ["to_q", "to_k", "to_v", "to_out.0"]
TEXT_TARGETS = ["q_proj", "k_proj", "v_proj", "out_proj"]


def add_lora(module: torch.nn.Module, rank: int, targets: list[str], name: str) -> None:
    from peft import LoraConfig

    cfg = LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian", target_modules=targets)
    module.add_adapter(cfg, adapter_name=name)


def set_active(module: torch.nn.Module, name: str) -> None:
    module.set_adapter(name)


def lora_state(module: torch.nn.Module, name: str) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items() if "lora_" in k and f".{name}." in k}


def load_lora_state(module: torch.nn.Module, state: dict[str, torch.Tensor]) -> None:
    missing = set(state) - set(module.state_dict())
    if missing:
        raise KeyError(f"LoRA keys not in model (rank or targets changed?): {sorted(missing)[:3]}")
    module.load_state_dict(state, strict=False)


def lora_params(module: torch.nn.Module, name: str) -> list[torch.nn.Parameter]:
    params = [p for n, p in module.named_parameters() if "lora_" in n and f".{name}." in n]
    for p in params:
        p.requires_grad_(True)
    return params


# --------------------------------------------------------------------------- data


def to_model_space(img: np.ndarray, mask: np.ndarray, res: int) -> tuple[torch.Tensor, torch.Tensor]:
    """RGB uint8 + bool mask -> pixel tensor in [-1, 1] (3,res,res) and mask (1,res,res)."""
    im = cv2.resize(img, (res, res), interpolation=cv2.INTER_AREA)
    mk = cv2.resize(mask.astype(np.uint8), (res, res), interpolation=cv2.INTER_NEAREST)
    x = torch.from_numpy(im).permute(2, 0, 1).float().div(127.5).sub(1)
    m = torch.from_numpy(mk).float()[None]
    return x, m


def random_boxes(res: int, rng: np.random.Generator, n_max: int = 3, lo: float = 0.1, hi: float = 0.5) -> torch.Tensor:
    m = torch.zeros(1, res, res)
    for _ in range(int(rng.integers(1, n_max + 1))):
        bw, bh = int(rng.uniform(lo, hi) * res), int(rng.uniform(lo, hi) * res)
        x0, y0 = int(rng.integers(0, res - bw + 1)), int(rng.integers(0, res - bh + 1))
        m[:, y0 : y0 + bh, x0 : x0 + bw] = 1
    return m


def _encode(comp: SDComponents, pixels: torch.Tensor, sample: bool = True) -> torch.Tensor:
    """VAE encode. Training samples the posterior; generation uses its mode so a seed fixes the output."""
    with torch.no_grad():
        dist = comp.vae.encode(pixels.to(comp.device, comp.dtype)).latent_dist
        return (dist.sample() if sample else dist.mode()) * comp.vae_scale


def _latent_mask(mask: torch.Tensor, size: int) -> torch.Tensor:
    # A latent cell counts as masked if any pixel under it is masked (max-pool), so thin scratches survive.
    return F.adaptive_max_pool2d(mask, (size, size))


def _target(comp: SDComponents, latents: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    pt = comp.train_scheduler.config.prediction_type
    if pt == "epsilon":
        return noise
    if pt == "v_prediction":
        return comp.train_scheduler.get_velocity(latents, noise, t)
    raise ValueError(f"Unsupported prediction type {pt}")


# --------------------------------------------------------------------------- training


@dataclass
class TrainConfig:
    objective: str  # "defectfill" | "magic"
    steps: int = 1500
    rank: int = 8
    lr_unet: float = 2e-4
    lr_text: float = 4e-5  # DefectFill only
    train_text_encoder: bool = True  # DefectFill only
    grad_accum: int = 1
    lambda_def: float = 0.5
    lambda_obj: float = 0.2
    lambda_attn: float = 0.05
    obj_background_weight: float = 0.3  # DefectFill alpha; the paper says only "alpha < 1"
    gpp_sigma: float = 1.0  # MAGIC
    flip: bool = True
    seed: int = 0
    gradient_checkpointing: bool = True
    log_every: int = 50


def _autocast(comp: SDComponents):
    if comp.device.type == "cuda" and comp.dtype == torch.float16:
        return torch.autocast("cuda", dtype=torch.float16)
    return torch.autocast("cpu", enabled=False)


def train_lora(
    comp: SDComponents,
    images: list[np.ndarray],
    masks: list[np.ndarray],
    object_name: str,
    adapter: str,
    cfg: TrainConfig,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, torch.Tensor]]:
    """Train one LoRA adapter for one defect type. Returns {"unet": state, "text": state}."""
    if cfg.objective not in ("defectfill", "magic"):
        raise ValueError(cfg.objective)
    rng = np.random.default_rng(cfg.seed)
    torch.manual_seed(cfg.seed)
    res = comp.resolution
    lat = res // comp.latent_factor
    data = [to_model_space(im, mk, res) for im, mk in zip(images, masks)]

    use_text = cfg.objective == "defectfill" and cfg.train_text_encoder
    add_lora(comp.unet, cfg.rank, UNET_TARGETS, adapter)
    set_active(comp.unet, adapter)
    groups = [{"params": lora_params(comp.unet, adapter), "lr": cfg.lr_unet}]
    if use_text:
        add_lora(comp.text_encoder, cfg.rank, TEXT_TARGETS, adapter)
        set_active(comp.text_encoder, adapter)
        groups.append({"params": lora_params(comp.text_encoder, adapter), "lr": cfg.lr_text})
    # LoRA weights train in fp32 even when the base model is fp16.
    for g in groups:
        for p in g["params"]:
            p.data = p.data.float()
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)

    if cfg.gradient_checkpointing:
        comp.unet.enable_gradient_checkpointing()
        if use_text:
            comp.text_encoder.gradient_checkpointing_enable()
    comp.unet.train()
    if use_text:
        comp.text_encoder.train()

    store = install_attention_capture(comp.unet) if cfg.objective == "defectfill" else None
    p_def = "A photo of sks"
    p_obj = f"A {object_name} with sks"
    p_magic = PLACEHOLDER
    if store is not None:
        tok_def = placeholder_positions(comp, p_def)

    sched = comp.train_scheduler
    T = sched.config.num_train_timesteps
    history = []
    for step in range(cfg.steps):
        x, m = data[int(rng.integers(len(data)))]
        if cfg.flip and rng.random() < 0.5:
            x, m = x.flip(-1), m.flip(-1)
        if cfg.flip and rng.random() < 0.5:
            x, m = x.flip(-2), m.flip(-2)
        x, m = x[None].to(comp.device), m[None].to(comp.device)
        latents = _encode(comp, x)
        m_lat = _latent_mask(m, lat).to(latents.dtype)
        masked_lat = _encode(comp, x * (1 - m))
        losses = {}
        with _autocast(comp):
            if cfg.objective == "defectfill":
                # L_def + L_attn on the defect prompt
                noise = torch.randn_like(latents)
                t = torch.randint(0, T, (1,), device=comp.device)
                noisy = sched.add_noise(latents, noise, t)
                store.reset(tok_def)
                enc = encode_text(comp, [p_def]) if use_text else encode_text(comp, [p_def]).detach()
                pred = comp.unet(torch.cat([noisy, m_lat, masked_lat], 1), t, encoder_hidden_states=enc).sample
                err = (pred.float() - _target(comp, latents, noise, t).float()) ** 2
                l_def = (err * m_lat).sum() / (m_lat.sum() * err.shape[1]).clamp_min(1.0)
                amap = attention_map(store, lat)
                l_attn = F.mse_loss(amap, m_lat[:, 0].float())
                # Recording must stay on through backward: gradient checkpointing re-runs the forward
                # and fails if the recomputation does different work from the original pass.
                ((cfg.lambda_def * l_def + cfg.lambda_attn * l_attn) / cfg.grad_accum).backward()
                store.reset(enabled=False)
                # L_obj on the object prompt, random boxes plus the defect
                box = torch.clamp(random_boxes(res, rng)[None].to(comp.device) + m, 0, 1)
                b_lat = _latent_mask(box, lat).to(latents.dtype)
                b_masked = _encode(comp, x * (1 - box))
                noise2 = torch.randn_like(latents)
                t2 = torch.randint(0, T, (1,), device=comp.device)
                noisy2 = sched.add_noise(latents, noise2, t2)
                enc2 = encode_text(comp, [p_obj]) if use_text else encode_text(comp, [p_obj]).detach()
                pred2 = comp.unet(torch.cat([noisy2, b_lat, b_masked], 1), t2, encoder_hidden_states=enc2).sample
                w = m_lat.float() + cfg.obj_background_weight * (1 - m_lat.float())
                err2 = (pred2.float() - _target(comp, latents, noise2, t2).float()) ** 2
                l_obj = (w * err2).sum() / (w.sum() * err2.shape[1])
                ((cfg.lambda_obj * l_obj) / cfg.grad_accum).backward()
                losses = {"def": l_def.item(), "obj": l_obj.item(), "attn": l_attn.item()}
            else:
                noise = torch.randn_like(latents)
                t = torch.randint(0, T, (1,), device=comp.device)
                noisy = sched.add_noise(latents, noise, t)
                with torch.no_grad():
                    enc = encode_text(comp, [p_magic])
                enc = enc + cfg.gpp_sigma * torch.randn_like(enc)  # Gaussian Prompt Perturbation
                pred = comp.unet(torch.cat([noisy, m_lat, masked_lat], 1), t, encoder_hidden_states=enc).sample
                loss = F.mse_loss(pred.float(), _target(comp, latents, noise, t).float())
                (loss / cfg.grad_accum).backward()
                losses = {"mse": loss.item()}
        if (step + 1) % cfg.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        history.append(losses)
        if cfg.log_every and (step + 1) % cfg.log_every == 0:
            recent = history[-cfg.log_every :]
            avg = {k: float(np.mean([h[k] for h in recent])) for k in recent[0]}
            log(f"[{adapter}] step {step + 1}/{cfg.steps} " + " ".join(f"{k}={v:.4f}" for k, v in avg.items()))

    comp.unet.eval()
    comp.text_encoder.eval()
    if cfg.gradient_checkpointing:
        comp.unet.disable_gradient_checkpointing()
        if use_text:
            comp.text_encoder.gradient_checkpointing_disable()
    if store is not None:
        from diffusers.models.attention_processor import AttnProcessor2_0

        comp.unet.set_attn_processor(AttnProcessor2_0())
    for g in groups:
        for p in g["params"]:
            p.requires_grad_(False)
    return {
        "unet": lora_state(comp.unet, adapter),
        "text": lora_state(comp.text_encoder, adapter) if use_text else {},
        "history": history,
    }


# --------------------------------------------------------------------------- sampling


def magic_guidance(step: int, n_steps: int, w_d: float = 7.5, w_m: float = 4.0) -> tuple[float, float]:
    """(scale inside the mask, scale outside). Inside ramps w_m -> w_d over denoising (cosine)."""
    p = step / max(1, n_steps - 1)
    w_in = w_m + (w_d - w_m) * 0.5 * (1 - math.cos(math.pi * p))
    return w_in, w_d


def constant_guidance(w: float) -> Callable[[int, int], tuple[float, float]]:
    return lambda step, n: (w, w)


@torch.no_grad()
def inpaint(
    comp: SDComponents,
    image: np.ndarray,
    mask: np.ndarray,
    prompt: str,
    n: int,
    steps: int,
    guidance: Callable[[int, int], tuple[float, float]],
    gpp_sigma: float = 0.0,
    seed: int = 0,
) -> list[np.ndarray]:
    """Generate ``n`` inpaintings at model resolution. Returns RGB uint8 arrays (res x res)."""
    res = comp.resolution
    lat = res // comp.latent_factor
    x, m = to_model_space(image, mask, res)
    x, m = x[None].to(comp.device), m[None].to(comp.device)
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    masked_lat = _encode(comp, x * (1 - m), sample=False).repeat(n, 1, 1, 1)
    m_lat = _latent_mask(m, lat).to(comp.dtype).repeat(n, 1, 1, 1)

    cond = encode_text(comp, [prompt] * n).to(comp.dtype)
    if gpp_sigma > 0:
        cond = cond + gpp_sigma * torch.randn(cond.shape, generator=g).to(comp.device, comp.dtype)
    uncond = encode_text(comp, [""] * n).to(comp.dtype)
    sched = comp.sample_scheduler
    sched.set_timesteps(steps, device=comp.device)
    latents = (torch.randn((n, 4, lat, lat), generator=g) * sched.init_noise_sigma).to(comp.device, comp.dtype)
    m_full = m_lat.float()
    for i, t in enumerate(sched.timesteps):
        inp = sched.scale_model_input(torch.cat([latents] * 2), t)
        inp = torch.cat([inp, torch.cat([m_lat] * 2), torch.cat([masked_lat] * 2)], 1)
        eps = comp.unet(inp, t, encoder_hidden_states=torch.cat([uncond, cond])).sample
        e_u, e_c = eps.float().chunk(2)
        w_in, w_out = guidance(i, len(sched.timesteps))
        w = w_in * m_full + w_out * (1 - m_full)  # per-pixel guidance map (SAG when w_in != w_out)
        e = e_u + w * (e_c - e_u)
        latents = sched.step(e.to(comp.dtype), t, latents).prev_sample
    imgs = comp.vae.decode(latents / comp.vae_scale).sample
    imgs = ((imgs.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return list(imgs)


def composite(host: np.ndarray, generated: np.ndarray, mask: np.ndarray, feather: int = 6, context: int = 6) -> np.ndarray:
    """Paste a model-resolution inpainting back into the full-resolution host, only around the mask."""
    from .copypaste import feather_alpha

    h, w = host.shape[:2]
    up = cv2.resize(generated, (w, h), interpolation=cv2.INTER_CUBIC)
    region = mask.astype(np.uint8)
    if context > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * context + 1, 2 * context + 1))
        region = cv2.dilate(region, k)
    a = feather_alpha(region > 0, feather)[..., None]
    return (a * up + (1 - a) * host.astype(np.float32)).round().clip(0, 255).astype(np.uint8)


def crop_box(mask: np.ndarray, pad: int = 16) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return max(0, xs.min() - pad), max(0, ys.min() - pad), min(w, xs.max() + 1 + pad), min(h, ys.max() + 1 + pad)


def load_lpips(device: torch.device) -> Callable[[np.ndarray, np.ndarray], float]:
    import lpips

    net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()

    @torch.no_grad()
    def dist(a: np.ndarray, b: np.ndarray) -> float:
        ta = torch.from_numpy(cv2.resize(a, (128, 128))).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(device)
        tb = torch.from_numpy(cv2.resize(b, (128, 128))).permute(2, 0, 1)[None].float().div(127.5).sub(1).to(device)
        return float(net(ta, tb).item())

    return dist


def save_adapter(path: str | Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"unet": state["unet"], "text": state.get("text", {})}, path)
