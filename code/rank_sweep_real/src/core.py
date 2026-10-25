"""Backend-agnostic core math for the real-direction rank sweep.

Everything here runs identically under NumpyBackend (CPU self-test, synthetic
weights) and TorchBackend (GPU, real model weights). No I/O, no HF, no CUDA
assumptions -- so the numbers the self-test prints are produced by exactly the
code that will run on the 5090.

Conventions
-----------
W   : [d_out, d_in] weight of a residual-stream *write* matrix (o_proj,
      down_proj). HF stores nn.Linear.weight as [out_features, in_features],
      and for these two modules out_features == d_model, so the refusal
      direction acts on the LEFT (row space).
D   : [d_model, K] column-orthonormal matrix of real activation directions,
      column 0 being the difference-in-means refusal direction.
dW  : the abliteration edit, dW = -alpha_k * D_k D_k^T W.
"""

NF4_CODES = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
]

EPS = 1e-12


# ---------------------------------------------------------------- directions

def orthonormalize(B, bk):
    """Column-orthonormalize [d, K] keeping column 0 fixed in direction."""
    return bk.qr_thin(B)


def norm_match_alpha(W, D, k, bk):
    """alpha_k such that ||dW_k||_F == ||dW_1||_F (rank-1 reference)."""
    Dk = D[:, :k]
    D1 = D[:, :1]
    num = bk.fnorm(bk.matmul(bk.transpose(D1), W))
    den = bk.fnorm(bk.matmul(bk.transpose(Dk), W))
    return num / max(den, EPS)


def build_edit(W, D, k, bk, alpha=None, norm_match=True):
    """Return (dW, alpha_k). dW = -alpha * D_k D_k^T W."""
    Dk = D[:, :k]
    if alpha is None:
        alpha = norm_match_alpha(W, D, k, bk) if norm_match else 1.0
    proj = bk.matmul(Dk, bk.matmul(bk.transpose(Dk), W))
    return -alpha * proj, alpha


# ---------------------------------------------------------------- quantizers

def int_rtn_per_row(W, bits, bk):
    """Symmetric per-output-channel round-to-nearest integer quantization."""
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    scale = bk.absmax_rows(W) / qmax + EPS
    q = bk.clip(bk.round(W / scale), qmin, qmax)
    return q * scale


def nf4_blockwise(W, bk, block=64, rows_per_chunk=256):
    """NF4 (QLoRA codebook), per-block absmax normalization.

    Chunked over output rows: the nearest-code search materializes a
    [n_elem/block, block, 16] distance tensor, which is 16x the weight for a
    7B down_proj. 256 rows at a time keeps that under ~1 GB.
    """
    d_out, d_in = W.shape[0], W.shape[1]
    table = bk.asarray(NF4_CODES)
    pieces = []
    for r0 in range(0, d_out, rows_per_chunk):
        chunk = W[r0:r0 + rows_per_chunk]
        n_rows = chunk.shape[0]
        pad = (-d_in) % block
        if pad:
            chunk = _hcat(chunk, bk.zeros((n_rows, pad)), bk)
        flat = bk.reshape(chunk, (-1, block))
        absmax = bk.absmax_rows(flat) + EPS
        x = flat / absmax
        idx = bk.argmin_last(bk.abs(bk.expand_last(x) - table))
        deq = bk.take(table, idx) * absmax
        out = bk.reshape(deq, (n_rows, d_in + pad))
        pieces.append(out[:, :d_in])
    return _vcat(pieces, bk)


def fp8_e4m3(W, bk):
    """Round to the nearest FP8 E4M3 value (per-tensor scaled to the format's
    dynamic range). Normal range only; subnormals are clamped, which matters
    for < 2^-6 * scale and is negligible at these weight magnitudes."""
    FP8_MAX = 448.0
    amax = bk.absmax_all(W)
    s = amax / FP8_MAX if amax > 0 else 1.0
    x = W / s
    # exponent of each element, clamped to E4M3 normal range [-6, 8]
    mag = bk.abs(x) + 1e-30
    e = _floor_log2(mag, bk)
    e = bk.clip(e, -6.0, 8.0)
    step = _pow2(e - 3.0, bk)          # 3 mantissa bits
    q = bk.round(x / step) * step
    q = bk.clip(q, -FP8_MAX, FP8_MAX)
    return q * s


def quantize(W, scheme, bk):
    s = scheme.lower()
    if s == "int4":
        return int_rtn_per_row(W, 4, bk)
    if s == "int8":
        return int_rtn_per_row(W, 8, bk)
    if s == "nf4":
        return nf4_blockwise(W, bk)
    if s == "fp8":
        return fp8_e4m3(W, bk)
    raise ValueError("unknown scheme: " + scheme)


# ---------------------------------------------------------------- metrics

def measure(W, dW, D, k, scheme, bk):
    """Weight-space metrics for one (module, k, scheme) cell.

    gain_pi      : <Q(W+dW)-Q(W), dW> / <dW, dW>.  1.0 == the quantizer
                   transmits the edit in full (perfect dithering).
    snr_global   : ||dW||_F / ||rho||_F,  rho = (Q(W+dW)-Q(W)) - dW
    snr_subspace : same ratio after projecting onto span(D_k) -- the only
                   component that can change the model's behavior along the
                   ablated directions.
    survive_sub  : ||D_k^T (Q(W+dW)-Q(W))||_F / ||D_k^T dW||_F
    """
    Wq = quantize(W, scheme, bk)
    Wq_abl = quantize(W + dW, scheme, bk)
    dWq = Wq_abl - Wq
    rho = dWq - dW

    n_dW = bk.fnorm(dW)
    n_rho = bk.fnorm(rho)
    pi = bk.inner(dWq, dW) / max(bk.inner(dW, dW), EPS)

    Dk_t = bk.transpose(D[:, :k])
    sub_dW = bk.fnorm(bk.matmul(Dk_t, dW))
    sub_rho = bk.fnorm(bk.matmul(Dk_t, rho))
    sub_dWq = bk.fnorm(bk.matmul(Dk_t, dWq))

    return {
        "scheme": scheme,
        "k": k,
        "norm_dW": n_dW,
        "norm_rho": n_rho,
        "gain_pi": pi,
        "snr_global": n_dW / max(n_rho, EPS),
        "snr_subspace": sub_dW / max(sub_rho, EPS),
        "survive_sub": sub_dWq / max(sub_dW, EPS),
        "norm_W": bk.fnorm(W),
        "rel_edit": n_dW / max(bk.fnorm(W), EPS),
    }


# ---------------------------------------------------------------- helpers

def _vcat(pieces, bk):
    if len(pieces) == 1:
        return pieces[0]
    if bk.name == "numpy":
        return bk.np.concatenate(pieces, axis=0)
    return bk.torch.cat(pieces, dim=0)


def _hcat(A, B, bk):
    if bk.name == "numpy":
        return bk.np.concatenate([A, B], axis=1)
    return bk.torch.cat([A, B], dim=1)


def _floor_log2(x, bk):
    if bk.name == "numpy":
        return bk.np.floor(bk.np.log2(x))
    return bk.torch.floor(bk.torch.log2(x))


def _pow2(e, bk):
    if bk.name == "numpy":
        return bk.np.power(2.0, e)
    return bk.torch.pow(bk.torch.as_tensor(2.0, device=e.device, dtype=e.dtype), e)
