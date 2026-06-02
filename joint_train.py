import torch
import numpy as np
from tqdm.auto import tqdm
from joint_model import JointGAINMissModel

def dagma_log_mse_loss(output, target, eps=1e-8):
    n, d = target.shape
    return 0.5 * d * torch.log((1.0 / n) * torch.sum((output - target) ** 2) + eps)

def rbf_kernel(X, Y):  # 基于RBF核衡量数据点之间的相似度
    batch_size, h_dim = X.shape
    X = X.to(Y.device)
    norms_x = X.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(X, X.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(Y, Y.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:  # 控制RBF核的扩展程度
        C = 2 * h_dim * 1.0 / scale
        res1 = torch.exp(-C * dists_x) + torch.exp(-C * dists_y)

        res1 = (1 - torch.eye(batch_size).to(X.device)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = torch.exp(-C * dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats += res1 - res2

    return stats / batch_size

def train_joint_gain_dagma(
    data,
    mask_missing,
    hidden_dims,
    device="cuda",
    hint_rate=0.1,
    alpha=10,
    lambda1=0.02,
    lambda2=0.005,
    mu=0.1,
    s=1.0,
    epochs=10000,
    lr_g=1e-3,
    lr_d=1e-3,
    lr_s=2e-4,
    beta_joint=0.5,
    pretrain_gain_epochs=0,
    warmup_s_epochs=0,
    update_s_every=2,
):
    model = JointGAINMissModel(
        data=data,
        mask_missing=mask_missing,
        hidden_dims=hidden_dims,
        device=device,
        hint_rate=hint_rate,
        alpha=alpha,
    ).to(device)

    opt_G = torch.optim.Adam(model.gain.G.parameters(), lr=lr_g)
    opt_D = torch.optim.Adam(model.gain.D.parameters(), lr=lr_d)
    opt_S = torch.optim.Adam(model.scm.parameters(), lr=lr_s, weight_decay=mu * lambda2)

    for epoch in tqdm(range(epochs), desc="Joint Training"):
        # ========= 1. update D =========
        model.gain.G.eval()
        model.gain.D.train()
        model.scm.eval()

        h = model.sample_hint()
        d_loss = model.gain.discriminator_loss(model.data, model.mask_gain, h)

        opt_D.zero_grad()
        d_loss.backward()
        opt_D.step()

        # ========= 2. update G =========
        model.gain.G.train()
        model.gain.D.eval()
        model.scm.eval()

        h = model.sample_hint()
        g_loss, mse_train, mse_test, x_imp = model.gain.generator_loss(model.data, model.mask_gain, h)
        x_recon = model.scm(x_imp)
        joint_rec = torch.mean((x_recon - x_imp) ** 2)

        # 弱联合 只加一个来自 DAGMA 的重建项
        g_joint_loss = g_loss + beta_joint * joint_rec

        opt_G.zero_grad()
        g_joint_loss.backward()
        opt_G.step()

        # ========= 3. update S =========
        model.gain.G.eval()
        model.gain.D.eval()
        model.scm.train()

        with torch.no_grad():
            h = model.sample_hint()
            x_imp, _, _, _ = model.forward_gain(h=h)

        x_recon = model.scm(x_imp)
        score = dagma_log_mse_loss(x_recon, x_imp)
        l1_reg = lambda1 * model.scm.fc1_l1_reg()
        h_val = model.scm.h_func(s)

        stage_len = max(epochs // 4, 1)
        stage_id = min(epoch // stage_len, 3)
        mu_t = mu * (0.7 ** stage_id)

        s_loss = mu_t * (score + l1_reg) + h_val

        opt_S.zero_grad()
        s_loss.backward()
        opt_S.step()

        if epoch % 1000 == 0:
            print(
                f"epoch={epoch} | "
                f"D={d_loss.item():.4f} | "
                f"G={g_joint_loss.item():.4f} | "
                f"S={s_loss.item():.4f}"
            )
        '''        
        # ========= 3. update S =========
        s_loss_value = None

        if epoch >= warmup_s_epochs and (epoch % update_s_every == 0):
            model.gain.G.eval()
            model.gain.D.eval()
            model.scm.train()

            with torch.no_grad():
                h = model.sample_hint()
                x_imp, _, _, _ = model.forward_gain(h=h)

            x_recon = model.scm(x_imp)
            score = dagma_log_mse_loss(x_recon, x_imp)
            l1_reg = lambda1 * model.scm.fc1_l1_reg()
            h_val = model.scm.h_func(s)

            stage_len = max(epochs // 4, 1)
            stage_id = min(epoch // stage_len, 3)
            mu_t = mu * (0.7 ** stage_id)
            
            s_loss = mu_t * (score + l1_reg) + h_val

            opt_S.zero_grad()
            s_loss.backward()
            opt_S.step()

            s_loss_value = s_loss.item()

        if epoch % 1000 == 0:
            s_str = f"{s_loss_value:.4f}" if s_loss_value is not None else "skip"
            print(
                f"epoch={epoch} | "
                f"D={d_loss.item():.4f} | "
                f"G={g_joint_loss.item():.4f} | "
                f"S={s_str}"
            )
        '''

    W_est = model.to_adj()
    if torch.is_tensor(W_est):
        W_est = W_est.detach().cpu().numpy()
    return model, W_est