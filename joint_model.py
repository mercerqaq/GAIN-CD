import torch
import torch.nn as nn
from dagma import DagmaMLP
from gain import GAIN

class JointGAINMissModel(nn.Module):
    def __init__(self, data, mask_missing, hidden_dims, device,
                 hint_rate=0.1, alpha=10):
        super().__init__()
        self.device = device
        self.data = data.float().to(device)

        # 外部你的 mask 若是 1=缺失, 0=观测，这里转成 GAIN 语义：1=观测, 0=缺失
        self.mask_missing = mask_missing.float().to(device)
        self.mask_gain = 1.0 - self.mask_missing

        self.d = data.shape[1]
        self.gain = GAIN(
            train_loader=None,
            n_features=self.d,
            hint_rate=hint_rate,
            alpha=alpha,
            device=device
        )
        self.scm = DagmaMLP(hidden_dims, device=device, bias=True)

    def sample_hint(self):
        B = self.gain._sample_b(shape=self.mask_gain.shape, p=self.gain.hint_rate).to(self.device)
        H = B * self.mask_gain + 0.5 * (1 - B)
        return H

    def forward_gain(self, z=None, h=None):
        if h is None:
            h = self.sample_hint()
        x_imp, x_imputed, x_new = self.gain.forward_impute(self.data, self.mask_gain, z)
        return x_imp, x_imputed, x_new, h

    def forward_joint(self, z=None, h=None):
        x_imp, x_imputed, x_new, h = self.forward_gain(z, h)
        x_recon = self.scm(x_imp)
        return x_imp, x_recon, x_imputed, x_new, h

    def to_adj(self):
        return self.scm.fc1_to_adj()