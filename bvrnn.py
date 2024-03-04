import torch
from torch import nn
from torch import Tensor
from typing import Tuple

class BVRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, rnn_dim, mean, std, log_sigma_init, activation='ELU'):
        super(BVRNN, self).__init__()

        activation = getattr(nn, activation)
        self.EPS = torch.finfo(torch.float).eps  # numerical logs

        self.mean = torch.nn.Parameter(torch.from_numpy(
            mean.astype('float32')), requires_grad=False)
        self.std = torch.nn.Parameter(torch.from_numpy(
            std.astype('float32')), requires_grad=False)

        self.log_sigma = torch.nn.Parameter(torch.Tensor([log_sigma_init]))
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.rnn_dim = rnn_dim

        # feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            activation(),
            nn.Linear(h_dim, h_dim),
            activation(),
            nn.Linear(h_dim, h_dim),
            activation())

        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            activation(),
            nn.Linear(h_dim, h_dim),
            activation(),
            nn.Linear(h_dim, h_dim),
            activation())

        # encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + rnn_dim, h_dim),
            activation(),
            nn.Linear(h_dim, h_dim),
            activation(),
            nn.Linear(h_dim, z_dim))


        self.prior = nn.Sequential(
            nn.Linear(rnn_dim, h_dim),
            activation(),
            nn.Linear(h_dim, h_dim),
            activation())

        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_logvar = nn.Linear(h_dim, z_dim)

        # decoder
        self.dec = nn.Sequential(nn.Linear(h_dim + rnn_dim, h_dim), activation(),
                                 nn.Linear(h_dim, h_dim), activation(),
                                 nn.Linear(h_dim, h_dim), activation(),
                                 nn.Linear(h_dim, x_dim))

        self.rnn = nn.GRU(2*h_dim, rnn_dim, num_layers=1, batch_first=True)


    def forward(self, y: Tensor, p_use_gen: float, detach_gen: bool) -> Tuple[Tensor, Tensor]:
        y = (y - self.mean[None, None, :]) / self.std[None, None, :]
        decoded = []
        kld_loss = []

        phi_x = self.phi_x(y[:, :, :])

        h = torch.zeros(1, y.size(0), self.rnn_dim, device=y.device)
        h2 = torch.zeros(1, y.size(0), self.rnn_dim, device=y.device)

        for t in range(y.size(1)):
            random_num = torch.rand([])
            phi_x_t = phi_x[:, t, :]

            # encoder
            if random_num < p_use_gen:
                enc_t = self.enc(torch.cat([phi_x_t, h2[-1, :, :]], 1))
                prior_t = self.prior(h2[-1, :, :])
            else:
                enc_t = self.enc(torch.cat([phi_x_t, h[-1, :, :]], 1))
                prior_t = self.prior(h2[-1, :, :])

            # encoder
            enc_mean_t = self.enc_mean(enc_t)
            enc_logvar_t = self.enc_logvar(enc_t)

            # prior
            prior_mean_t = self.prior_mean(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(enc_mean_t, torch.exp(0.5 * enc_logvar_t))
            phi_z_t = self.phi_z(z_t)
            if random_num < p_use_gen:
                dec_mean_t = self.dec(torch.cat([phi_z_t, h2[-1, :, :]], 1))
            else:
                dec_mean_t = self.dec(torch.cat([phi_z_t, h[-1, :, :]], 1))

            if detach_gen:
                phi_x_t_gen = self.phi_x(
                    (dec_mean_t.detach() - self.mean[None, :]) / self.std[None, :])
            else:
                phi_x_t_gen = self.phi_x(
                    (dec_mean_t - self.mean[None, :]) / self.std[None, :])

            # recurrence
            if p_use_gen < 1:
                _, h = self.rnn(
                    torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(1), h)
            if p_use_gen > 0:
                _, h2 = self.rnn(
                    torch.cat([phi_x_t_gen, phi_z_t], 1).unsqueeze(1), h2)

            # computing losses
            kld_loss.append(0.5 * torch.mean(torch.sum(prior_logvar_t - enc_logvar_t + torch.div((enc_logvar_t.exp() + (enc_mean_t - prior_mean_t).pow(2)), prior_logvar_t.exp()), dim=1), dim=0))

            decoded.append(dec_mean_t)

        return torch.stack(decoded).permute(1, 0, 2), torch.mean(torch.stack(kld_loss))


    def sample(self, seq_len: float) -> Tensor:
        sample = torch.zeros((1, seq_len, self.x_dim), device='cuda')

        h = torch.zeros(1, 1, self.rnn_dim, device='cuda')
        for t in range(seq_len):

            prior_t = self.prior(h[-1, :, :])
            prior_mean_t = self.prior_mean(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)

            # sampling and reparameterization
            z_t = self._reparameterized_sample(prior_mean_t, torch.exp(0.5 * prior_logvar_t))
            phi_z_t = self.phi_z(z_t)

            # decoder
            dec_mean_t = self.dec(torch.cat([phi_z_t, h[-1, :, :]], 1))

            phi_x_t = self.phi_x(
                (dec_mean_t - self.mean[None, :]) / self.std[None, :])

            # recurrence
            _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(1), h)

            sample[:, t, :] = dec_mean_t.data

        return sample

    def _reparameterized_sample(self, mean: Tensor, std: Tensor) -> Tensor:
        """using std to sample"""
        eps = torch.empty(size=std.size(), device=mean.device,
                          dtype=torch.float).normal_()
        return eps.mul(std).add_(mean)
