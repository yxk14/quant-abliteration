"""Thin array-backend adapter so the same math runs under numpy (CPU self-test)
and torch (GPU, real weights). Only the ops core.py needs are exposed."""


class NumpyBackend:
    name = "numpy"

    def __init__(self):
        import numpy as np
        self.np = np

    def asarray(self, x, dtype="f8"):
        return self.np.asarray(x, dtype=dtype)

    def zeros(self, shape):
        return self.np.zeros(shape, dtype="f8")

    def matmul(self, a, b):
        return a @ b

    def transpose(self, a):
        return a.T

    def fnorm(self, a):
        return float(self.np.linalg.norm(a))

    def inner(self, a, b):
        return float((a * b).sum())

    def absmax_rows(self, a):
        return self.np.abs(a).max(axis=1, keepdims=True)

    def absmax_all(self, a):
        return float(self.np.abs(a).max())

    def round(self, a):
        return self.np.round(a)

    def clip(self, a, lo, hi):
        return self.np.clip(a, lo, hi)

    def reshape(self, a, shape):
        return a.reshape(shape)

    def where(self, cond, a, b):
        return self.np.where(cond, a, b)

    def sign(self, a):
        return self.np.sign(a)

    def abs(self, a):
        return self.np.abs(a)

    def argmin_last(self, a):
        return self.np.argmin(a, axis=-1)

    def take(self, table, idx):
        return table[idx]

    def expand_last(self, a):
        return a[..., None]

    def qr_thin(self, a):
        q, _ = self.np.linalg.qr(a)
        return q

    def svd_u(self, a, k):
        u, s, _ = self.np.linalg.svd(a, full_matrices=False)
        return u[:, :k], s[:k]

    def to_cpu_list(self, a):
        return self.np.asarray(a).tolist()

    def maximum(self, a, b):
        return self.np.maximum(a, b)


class TorchBackend:
    name = "torch"

    def __init__(self, device="cuda", dtype=None):
        import torch
        self.torch = torch
        self.device = device
        self.dtype = dtype or torch.float32

    def asarray(self, x, dtype=None):
        t = self.torch
        if isinstance(x, t.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return t.as_tensor(x, device=self.device, dtype=self.dtype)

    def zeros(self, shape):
        return self.torch.zeros(shape, device=self.device, dtype=self.dtype)

    def matmul(self, a, b):
        return a @ b

    def transpose(self, a):
        return a.transpose(-2, -1)

    def fnorm(self, a):
        return float(self.torch.linalg.vector_norm(a.float()))

    def inner(self, a, b):
        return float((a.float() * b.float()).sum())

    def absmax_rows(self, a):
        return a.abs().amax(dim=1, keepdim=True)

    def absmax_all(self, a):
        return float(a.abs().amax())

    def round(self, a):
        return self.torch.round(a)

    def clip(self, a, lo, hi):
        return self.torch.clamp(a, lo, hi)

    def reshape(self, a, shape):
        return a.reshape(shape)

    def where(self, cond, a, b):
        return self.torch.where(cond, a, b)

    def sign(self, a):
        return self.torch.sign(a)

    def abs(self, a):
        return a.abs()

    def argmin_last(self, a):
        return self.torch.argmin(a, dim=-1)

    def take(self, table, idx):
        return table[idx]

    def expand_last(self, a):
        return a.unsqueeze(-1)

    def qr_thin(self, a):
        q, _ = self.torch.linalg.qr(a.float())
        return q.to(self.dtype)

    def svd_u(self, a, k):
        u, s, _ = self.torch.linalg.svd(a.float(), full_matrices=False)
        return u[:, :k].to(self.dtype), s[:k]

    def to_cpu_list(self, a):
        return a.detach().float().cpu().tolist()

    def maximum(self, a, b):
        return self.torch.maximum(a, b)


def get_backend(which, device="cuda"):
    if which == "numpy":
        return NumpyBackend()
    return TorchBackend(device=device)
