import torch
from torch.nn import functional as F
from models.soft_dtw import SoftDTW

sdtw = SoftDTW(use_cuda=False, gamma=0.01, warp=134.4)


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l


def get_sdtw_kl_matrix(z_p, logs_q, m_p, logs_p):
    """
    returns kl matrix with shape [b, t_tp, t_tq]
    z_p, logs_q: [b, h, t_tq]
    m_p, logs_p: [b, h, t_tp]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()

    t_tp, t_tq = m_p.size(-1), z_p.size(-1)
    b, h, t_tp = m_p.shape

    # for memory testing
    # return torch.abs(z_p.mean(dim=1)[:, None, :] - m_p.mean(dim=1)[:, :, None])

    # z_p = z_p.transpose(0, 1)
    # logs_q = logs_q.transpose(0, 1)
    # m_p = m_p.transpose(0, 1)
    # logs_p = logs_p.transpose(0, 1)

    kls = torch.zeros((b, t_tp, t_tq), dtype=z_p.dtype, device=z_p.device)
    for i in range(h):
        logs_p_, m_p_, logs_q_, z_p_ = (
            logs_p[:, i, :, None],
            m_p[:, i, :, None],
            logs_q[:, i, None, :],
            z_p[:, i, None, :],
        )
        kl = logs_p_ - logs_q_ - 0.5  # [b, t_tp, t_tq]
        kl += 0.5 * ((z_p_ - m_p_) ** 2) * torch.exp(-2.0 * logs_p_)
        kls += kl
    return kls

    kl = logs_p[:, :, :, None] - logs_q[:, :, None, :] - 0.5  # p, q
    kl += (
        0.5
        * ((z_p[:, :, None, :] - m_p[:, :, :, None]) ** 2)
        * torch.exp(-2.0 * logs_p[:, :, :, None])
    )

    kl = kl.sum(dim=1)
    return kl


import torch.utils.checkpoint


def kl_loss_dtw(z_p, logs_q, m_p, logs_p, p_mask, q_mask):
    INF = 1e5

    kl = get_sdtw_kl_matrix(z_p, logs_q, m_p, logs_p)  # [b t_tp t_tq]
    kl = torch.nn.functional.pad(kl, (0, 1, 0, 1), "constant", 0)
    p_mask = torch.nn.functional.pad(p_mask, (0, 1), "constant", 0)
    q_mask = torch.nn.functional.pad(q_mask, (0, 1), "constant", 0)

    kl.masked_fill_(p_mask[:, :, None].bool() ^ q_mask[:, None, :].bool(), INF)
    kl.masked_fill_((~p_mask[:, :, None].bool()) & (~q_mask[:, None, :].bool()), 0)
    res = sdtw(kl).sum() / p_mask.sum()
    return res


if __name__ == "__main__":
    kl = torch.rand(4, 100, 100)
    kl[:, 50:, :] = 1e4
    kl[:, :, 50:] = 1e4
    kl[:, 50:, 50:] = 0

    print(kl)
    print(sdtw(kl).mean() / 50)
