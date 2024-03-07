import torch
from torch import nn
from torch import Tensor, LongTensor
from typing import Tuple

class BVRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, mean_std_mel, log_sigma_init, variableBit = False):
        super(BVRNN, self).__init__()
  
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


        self.mean_mel = torch.nn.Parameter(torch.from_numpy(mean_std_mel[0].astype('float32')), requires_grad=False)
        self.std_mel = torch.nn.Parameter(torch.from_numpy(mean_std_mel[1].astype('float32')), requires_grad=False)

        self.log_sigma = torch.nn.Parameter(torch.Tensor([log_sigma_init]))
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.varBit = variableBit
        if variableBit:
            var_dim = z_dim
        else :
            var_dim = 0

        #feature-extracting transformations
        self.phi_x = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU())
        
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim + var_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU())
        
        #encoder
        self.enc = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, z_dim), nn.Sigmoid())
    

        self.prior = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, z_dim), nn.Sigmoid())
        

        #decoder
        self.dec = nn.Sequential(nn.Linear(2*h_dim, h_dim), nn.ELU(),
                                 nn.Linear(h_dim, h_dim), nn.ELU(),
                                 nn.Linear(h_dim, h_dim), nn.ELU(),
                                 nn.Linear(h_dim, x_dim))


        self.rnn = nn.GRU(2*h_dim, h_dim, num_layers=1, batch_first=True)


    def forward(self, y: Tensor, p_use_gen: float, greedy: bool, varBitrate: Tensor)->Tuple[Tensor, Tensor]:
        
        y = (y - self.mean_mel[None, None, :]) / self.std_mel[None, None, :]

        all_dec_mean = []
        kld_loss = []

        phi_x = self.phi_x(y)

        h = torch.zeros(1, y.size(0), self.h_dim, device=self.device)
        h2 = torch.zeros(1, y.size(0), self.h_dim, device=self.device)
        if self.varBit:
            bit_cond_helper = torch.arange(0, self.z_dim, 1).to(y.device)
            bit_cond = ((varBitrate[:,:,None] - 1) == bit_cond_helper).float()
            bit_mask = varBitrate[:,:,None] > bit_cond_helper[None,None,:]
        else:
            bit_cond = torch.zeros((0,0,0), device=y.device)
            bit_cond_helper = torch.zeros((0,0,0), device=y.device)
            bit_mask = torch.zeros((0,0,0), device=y.device)

        for t in range(y.size(1)):
            random_num = torch.rand([])
            phi_x_t = phi_x[:, t, :]

            if random_num < p_use_gen:
                enc_t = self.enc(torch.cat([phi_x_t, h2[-1, :, :]], 1))
                prior_t = self.prior(h2[-1, :, :])
            else:
                enc_t = self.enc(torch.cat([phi_x_t, h[-1, :, :]], 1))
                prior_t = self.prior(h[-1, :, :])

            #sampling and reparameterization
            if greedy:
                z_t = torch.round(enc_t.detach()) - enc_t.detach() + enc_t
            else:
                z_t = torch.round(torch.rand_like(enc_t).to(enc_t.device) - 0.5 + enc_t.detach()) - enc_t.detach() + enc_t

            #variable bitrate
            if self.varBit:
                z_t[torch.logical_not(bit_mask[:,t,:])] = 0.5
                z_t_var = torch.cat((z_t, bit_cond[:,t,:]), dim=-1)
                enc_t = enc_t * bit_mask[:,t,:].float() + 0.5 * (1 - bit_mask[:,t,:].float())
                phi_z_t = self.phi_z(z_t_var)
            else:
                phi_z_t = self.phi_z(z_t)
            
            
            if random_num < p_use_gen:
                dec_t = self.dec(torch.cat([phi_z_t, h2[-1, :, :]], 1))
            else:
                dec_t = self.dec(torch.cat([phi_z_t, h[-1, :, :]], 1))
            
            phi_x_t_gen = self.phi_x((dec_t - self.mean_mel[None, :]) / self.std_mel[None, :])

            #recurrence
            if p_use_gen < 1:
                _, h = self.rnn(torch.cat([phi_x_t, phi_z_t], 1).unsqueeze(1), h)
            if p_use_gen > 0:
                _, h2 = self.rnn(torch.cat([phi_x_t_gen, phi_z_t], 1).unsqueeze(1), h2)

                

            #computing losses
            kld_loss.append(torch.mean(torch.sum(enc_t * (torch.log(enc_t + 1e-4) - torch.log(prior_t + 1e-4)) +
                            (1-enc_t) * (torch.log((1-enc_t) + 1e-4) - torch.log((1-prior_t) + 1e-4)), -1)))

            all_dec_mean.append(dec_t)

        return torch.stack(all_dec_mean).permute(1, 0, 2), torch.mean(torch.stack(kld_loss))