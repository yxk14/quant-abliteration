"""Simulated (quantize-dequantize) weight quantizers.

Every function maps a float tensor to the SAME dtype, having passed through
the representable set of the target format. This isolates representational
error from kernel behaviour, and lets INT16/FP8 be measured on GPUs with no
native kernel for them (e.g. Colab's A100).
"""
import torch

# --------------------------------------------------------------- integer
def quant_int(W: torch.Tensor, bits: int, per_channel: bool = True) -> torch.Tensor:
    """Symmetric affine integer quantization, per output channel by default."""
    qmax = 2 ** (bits - 1) - 1
    dim = tuple(range(1, W.ndim)) if per_channel else None
    amax = W.abs().amax(dim=dim, keepdim=True) if per_channel else W.abs().amax()
    scale = (amax / qmax).clamp(min=1e-12)
    return (torch.round(W / scale).clamp(-qmax - 1, qmax) * scale).to(W.dtype)

def quant_int_group(W: torch.Tensor, bits: int, group: int = 128) -> torch.Tensor:
    """Group-wise symmetric integer RTN: one scale per `group` consecutive weights
    along the input dimension. This is what deployed integer kernels actually do
    (GPTQ/AWQ/GGUF all group); plain per-channel RTN is a strawman by comparison,
    because one outlier sets the scale for a whole 3584-wide row."""
    qmax = 2 ** (bits - 1) - 1
    shp = W.shape
    flat = W.reshape(-1)
    pad = (-flat.numel()) % group
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    g = flat.view(-1, group)
    amax = g.abs().amax(dim=1, keepdim=True)
    scale = (amax / qmax).clamp(min=1e-12)
    out = (torch.round(g / scale).clamp(-qmax - 1, qmax) * scale).view(-1)
    return out[:W.numel()].view(shp).to(W.dtype)


# ------------------------------------------------------------ floating point
def _fp_roundtrip(W, exp_bits, man_bits, per_channel=True):
    """Generic float format with given exponent/mantissa widths, with a
    per-channel scale so the format's limited range is used efficiently."""
    bias = 2 ** (exp_bits - 1) - 1
    max_exp = bias                      # reserve top exponent
    fmax = (2 - 2 ** -man_bits) * 2 ** max_exp
    dim = tuple(range(1, W.ndim)) if per_channel else None
    amax = W.abs().amax(dim=dim, keepdim=True) if per_channel else W.abs().amax()
    scale = (amax / fmax).clamp(min=1e-12)
    X = (W / scale).clamp(-fmax, fmax)
    e = torch.floor(torch.log2(X.abs().clamp(min=1e-30)))
    e = e.clamp(min=-bias + 1, max=max_exp)
    step = torch.pow(2.0, e - man_bits)
    return (torch.round(X / step) * step * scale).to(W.dtype)

def quant_fp16(W):  return W.to(torch.float16).to(W.dtype)
def quant_fp8(W):   return _fp_roundtrip(W, exp_bits=4, man_bits=3)   # E4M3
# Note: this is an E4M3-shaped simulation (4 exponent bits, 3 mantissa bits with
# per-channel scaling), not a bit-exact OCP E4M3 round-trip -- the OCP format's
# extended max (448) and special encodings are not reproduced. On Hopper/Ada,
# cross-check against torch.float8_e4m3fn; the per-channel scale absorbs most of
# the difference, but report which was used.
def quant_fp4(W):   return quant_nf4(W)                               # NF4 codebook

# NF4: the information-theoretically-motivated 4-bit codebook from QLoRA.
_NF4 = torch.tensor([
 -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844444, -0.1848917, -0.0910390, 0.0,
  0.0795803, 0.1609302, 0.2461123, 0.3379152, 0.4407098, 0.5626170, 0.7229568, 1.0])

def quant_nf4(W: torch.Tensor, block: int = 64) -> torch.Tensor:
    """Block-wise NF4 with absmax scaling (as in bitsandbytes)."""
    shp, flat = W.shape, W.reshape(-1)
    pad = (-flat.numel()) % block
    if pad: flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.view(-1, block)
    amax = blocks.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    normed = blocks / amax
    cb = _NF4.to(W.device, W.dtype)
    # Chunked codebook search. The naive form materialises
    # [n_blocks, block, 16], which is ~1 GB for one 4096x4096 matrix.
    outs, CH = [], 8192
    for i in range(0, normed.shape[0], CH):
        chunk = normed[i:i+CH]
        idx = (chunk.unsqueeze(-1) - cb).abs().argmin(dim=-1)
        outs.append(cb[idx] * amax[i:i+CH])
    out = torch.cat(outs, dim=0).reshape(-1)
    return out[:W.numel()].view(shp).to(W.dtype)

# ------------------------------------------------- native Blackwell formats
# The RTX 5090 has hardware FP8 (E4M3/E5M2) and FP4. Running these ALONGSIDE
# the simulated versions does two things: validates the simulation against real
# silicon, and -- by comparing NF4 (codebook) against NVFP4 (hardware float) at
# identical bit width -- separates whether it is the CODEBOOK or the BIT WIDTH
# that breaks abliterated weights.
def _native(dtype_name, W, per_channel=True):
    dt = getattr(torch, dtype_name, None)
    if dt is None:
        raise RuntimeError(f"torch.{dtype_name} unavailable -- needs a Blackwell/"
                           f"Hopper GPU and a recent torch build")
    dim = tuple(range(1, W.ndim)) if per_channel else None
    amax = W.abs().amax(dim=dim, keepdim=True) if per_channel else W.abs().amax()
    # scale into the format's representable range before casting
    finfo_max = torch.finfo(dt).max if hasattr(torch.finfo(dt), "max") else 448.0
    scale = (amax / finfo_max).clamp(min=1e-12)
    return ((W / scale).to(dt).to(W.dtype) * scale)

def quant_fp8_native_e4m3(W): return _native("float8_e4m3fn", W)
def quant_fp8_native_e5m2(W): return _native("float8_e5m2", W)
def quant_fp4_native(W):      return _native("float4_e2m1fn_x2", W)

def native_available():
    """Which native formats this GPU + torch build actually supports."""
    out = {}
    for k, n in [("FP8E4M3", "float8_e4m3fn"), ("FP8E5M2", "float8_e5m2"),
                 ("FP4NV", "float4_e2m1fn_x2")]:
        try:
            if not hasattr(torch, n): out[k] = False; continue
            x = torch.randn(64, 64, device="cuda" if torch.cuda.is_available() else "cpu")
            _native(n, x); out[k] = True
        except Exception:
            out[k] = False
    return out


QUANTIZERS = {
    ("FP8N", 8): quant_fp8_native_e4m3,
    ("FP8E5", 8): quant_fp8_native_e5m2,
    ("FP4N", 4): quant_fp4_native,
    ("FP", 16): quant_fp16,
    ("FP",  8): quant_fp8,
    ("FP",  4): quant_fp4,
    ("INT",16): lambda W: quant_int(W, 16),
    ("INT", 8): lambda W: quant_int(W, 8),
    ("INT", 4): lambda W: quant_int(W, 4),
}
def get_quantizer(fmt: str, width: int):
    if (fmt, width) in QUANTIZERS: return QUANTIZERS[(fmt, width)]
    if fmt == "INTG": return lambda W: quant_int_group(W, width, 128)
    if fmt == "INT": return lambda W: quant_int(W, width)
    raise KeyError(f"no quantizer for {fmt}{width}")
