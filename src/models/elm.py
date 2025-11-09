import torch
import torch.nn as nn

class RandomFeatureELM(nn.Module):
    """
    Random features + ridge closed-form.
    Fit: H = phi(X W + b), beta = (H^T H + λI)^{-1} H^T Y
    Predict: Yhat = H beta
    """
    def __init__(self, in_dim, out_dim, hidden=1024, act="tanh", ridge_lambda=1e-2, seed=0, device="cpu"):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W = nn.Parameter(torch.randn(in_dim, hidden, generator=g)*0.5, requires_grad=False)
        self.b = nn.Parameter(torch.randn(hidden, generator=g)*0.5, requires_grad=False)
        self.act = act
        self.ridge = ridge_lambda
        self.beta = None
        self.out_dim = out_dim
        self.device = device

    def _phi(self, XH):
        if self.act == "tanh":
            return torch.tanh(XH)
        elif self.act == "relu":
            return torch.relu(XH)
        else:
            return XH

    def fit(self, X, Y):
        # X: [N, in_dim], Y: [N, out_dim]
        X = X.to(self.device)
        Y = Y.to(self.device)
        H = self._phi(X @ self.W + self.b)    # [N, hidden]
        # ridge solve
        HtH = H.T @ H
        lamI = self.ridge * torch.eye(HtH.shape[0], device=self.device)
        self.beta = torch.linalg.solve(HtH + lamI, H.T @ Y)  # [hidden, out_dim]

    def predict(self, X):
        X = X.to(self.device)
        H = self._phi(X @ self.W + self.b)
        return H @ self.beta

