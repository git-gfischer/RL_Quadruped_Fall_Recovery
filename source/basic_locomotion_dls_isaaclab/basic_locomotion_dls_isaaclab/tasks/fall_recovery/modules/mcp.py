# Mass Contact Predictor Network
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class MCPDims:
    o_dim: int = 42
    m_dim: int = 4
    c_dim: int = 13
    z_dim: int = 16
    H: int = 5  # history length

class MCPDataset:
    def __init__(self, max_size=200_000, device="cuda:0"):
        self.device = device
        self.max_size = int(max_size)
        self.ptr = 0
        self.full = False
        self.buffers = None  # lazy init
        self.size_cached = 0

    def _allocate(self, batch):
        # batch values are (N, D...). We store (max_size, D...)
        self.buffers = {}
        for k, v in batch.items():
            shape = (self.max_size, *v.shape[1:])
            self.buffers[k] = torch.empty(shape, device=self.device, dtype=v.dtype)

    def add(self, o_hist, o_next, m_true, c_true):
        """
        Add a mini-batch of samples (one per env): tensors are shaped (N, D).
        Supports circular wrap-around.
        """
        batch = {"o_hist": o_hist, "o_next": o_next, "m_true": m_true, "c_true": c_true}
        # Move to our device (no copy if already there)
        for k in batch:
            batch[k] = batch[k].to(self.device)

        if self.buffers is None:
            self._allocate(batch)

        N = o_hist.shape[0]
        N = int(N)

        # write in up to 2 slices (head then wrapped tail)
        end = self.ptr + N
        if end <= self.max_size:
            sl = slice(self.ptr, end)
            assert self.buffers is not None
            for k, v in batch.items():
                self.buffers[k][sl] = v
        else:
            first = self.max_size - self.ptr
            sl1 = slice(self.ptr, self.max_size)
            sl2 = slice(0, end % self.max_size)
            assert self.buffers is not None
            for k, v in batch.items():
                self.buffers[k][sl1] = v[:first]
                self.buffers[k][sl2] = v[first:]

        self.ptr = end % self.max_size
        self.full = self.full or (end >= self.max_size)
        self.size_cached = self.max_size if self.full else max(self.size_cached, end)

    def __len__(self):
        return self.size_cached

    def sample(self, batch_size):
        size = len(self)
        if size == 0:
            raise RuntimeError("MCPDataset is empty; cannot sample.")
        bs = min(int(batch_size), size)
        idx = torch.randint(0, size, (bs,), device=self.device)
        assert self.buffers is not None
        return {k: v.index_select(0, idx) for k, v in self.buffers.items()}

class Encoder(nn.Module):
    def __init__(self, dims: MCPDims):
        super().__init__()
        self.dims = dims
        in_dim = dims.o_dim * dims.H
        hid = 256
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
        )
        # heads
        self.m_head = nn.Linear(hid, dims.m_dim)
        self.c_head = nn.Linear(hid, dims.c_dim)
        self.z_mu = nn.Linear(hid, dims.z_dim)
        self.z_logvar = nn.Linear(hid, dims.z_dim)

    def forward(self, o_hist):
        h = self.net(o_hist)
        m_hat = self.m_head(h)
        c_hat_logits = self.c_head(h)
        z_mu = self.z_mu(h)
        z_logvar = self.z_logvar(h)
        return m_hat, c_hat_logits, z_mu, z_logvar

class Decoder(nn.Module):
    def __init__(self, dims: MCPDims):
        super().__init__()
        in_dim = dims.m_dim + dims.c_dim + dims.z_dim
        hid = 256
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.ReLU(),
            nn.Linear(hid, dims.o_dim)
        )

    def forward(self, m_hat, c_hat, z):
        x = torch.cat([m_hat, c_hat, z], dim=-1)
        return self.net(x)

class MCP(nn.Module):
    def __init__(self, dims=MCPDims()):
        super().__init__()
        self.dims = dims
        self.encoder = Encoder(dims)
        self.decoder = Decoder(dims)
        self.dataset = None  # assigned by env

    def forward(self, o_hist):
        m_hat, c_logits, z_mu, z_logvar = self.encoder(o_hist)
        # reparameterize
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mu + eps * std
        c_hat = torch.sigmoid(c_logits)
        return m_hat, c_hat, z, (z_mu, z_logvar)

    @staticmethod
    def kl_loss(z_mu, z_logvar):
        # D_KL(N(mu, sigma) || N(0, I))
        return 0.5 * torch.sum(torch.exp(z_logvar) + z_mu**2 - 1.0 - z_logvar, dim=-1)

    def train_step(self, batch, weights, optim):
        # Escape any outer inference/no_grad
        with torch.inference_mode(False):
            with torch.enable_grad():
                self.train(True)

                device = next(self.parameters()).device
                dtype  = next(self.parameters()).dtype

                o_hist = batch["o_hist"].to(device=device, dtype=dtype)
                o_next = batch["o_next"].to(device=device, dtype=dtype)
                m_true = batch["m_true"].to(device=device, dtype=dtype)
                c_true = batch["c_true"].to(device=device, dtype=dtype).clamp(0.0, 1.0)

                lam_m   = torch.as_tensor(weights.get("lambda_m",   1.0),  device=device, dtype=dtype)
                lam_c   = torch.as_tensor(weights.get("lambda_c",   1.0),  device=device, dtype=dtype)
                lam_rec = torch.as_tensor(weights.get("lambda_rec", 1.0),  device=device, dtype=dtype)
                lam_kl  = torch.as_tensor(weights.get("lambda_kl",  1e-3), device=device, dtype=dtype)

                # Explicit forward (no no_grad here)
                m_hat, c_logits, z_mu, z_logvar = self.encoder(o_hist)
                std = torch.exp(0.5 * z_logvar)
                z   = z_mu + torch.randn_like(std) * std
                c_hat = torch.sigmoid(c_logits)
                o_pred = self.decoder(m_hat, c_hat, z)

                lm   = F.mse_loss(m_hat,  m_true)
                lc   = F.binary_cross_entropy(c_hat, c_true)
                lrec = F.mse_loss(o_pred, o_next)
                lkl  = self.kl_loss(z_mu, z_logvar).mean()
                loss = lam_m * lm + lam_c * lc + lam_rec * lrec + lam_kl * lkl

                if not loss.requires_grad:
                    raise RuntimeError("MCP loss still has no grad (check for .detach() or @torch.no_grad on encoder/decoder).")

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                return {
                    "loss": loss.detach(), "lm": lm.detach(), "lc": lc.detach(),
                    "lrec": lrec.detach(), "lkl": lkl.detach(),
                }