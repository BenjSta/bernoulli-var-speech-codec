# This code loosely follows the implementations in
# https://github.com/emited/VariationalRecurrentNeuralNetwork
# and 
# https://github.com/XiaoyuBIE1994/DVAE
#
# (Benjamin Stahl, IEM Graz, 2024)

import torch
from torch import nn
from torch import Tensor
from typing import Tuple

class BVRNN(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, mean_std_mel, log_sigma_init, variableBit = False):
        '''
        Bernoulli-valued Variational Recurrent Neural Network

        x_dim: dimensionality of the input
        h_dim: dimensionality of the hidden state
        z_dim: dimensionality of the latent variable
        mean_std_mel: mean and std of the mel spectrogram
        log_sigma_init: initial value of the log of the scale of the data noise (to balance the KLD and the reconstruction loss)
        variableBit: whether to use variable bitrate
        '''
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
            nn.Linear(z_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU(),
            nn.Linear(h_dim, h_dim),
            nn.ELU())
        
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
        '''
        y: input tensor, shape (batch, frames, x_dim)
        p_use_gen: probability of using the generator
        greedy: whether to sample (False) or just round (True) the binary probabilities
        varBitrate: tensor of the variable bitrate in bits per frame, shape (batch, frames), will be ignored if variableBit is False

        returns: reconstructed tensor (shape (batch, frames, x_dim)), KLD loss (scalar)
        '''
        y = (y - self.mean_mel[None, None, :]) / self.std_mel[None, None, :]

        all_dec_mean = []
        kld_loss = []

        phi_x = self.phi_x(y)

        h = torch.zeros(1, y.size(0), self.h_dim, device=self.device)
        h2 = torch.zeros(1, y.size(0), self.h_dim, device=self.device)
        if self.varBit:
            bit_cond_helper = torch.arange(0, self.z_dim, 1).to(y.device)
            bit_mask = varBitrate[:,:,None] > bit_cond_helper[None,None,:]
        else:
            bit_mask = torch.zeros((0,0,0), device=y.device)

        for t in range(y.size(1)):
            random_num = torch.rand([])
            phi_x_t = phi_x[:, t, :]
            # if self.varBit:

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

            if self.varBit:
                z_t = z_t * bit_mask[:,t,:].float() + 0.5 * (1-bit_mask[:,t,:].float())
                
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

                
            kld_elem = enc_t * (torch.log(torch.clip(enc_t, 1e-3)) - torch.log(torch.clip(prior_t, 1e-3))) + \
                            (1-enc_t) * (torch.log(torch.clip((1-enc_t), 1e-3)) - torch.log(torch.clip((1-prior_t), 1e-3)))
            
            #variable bitrate
            if self.varBit:
                #computing losses
                kld_loss.append(torch.mean(torch.sum(kld_elem * bit_mask[:,t,:].float(), -1)))
            else:
                kld_loss.append(torch.mean(torch.sum(kld_elem, -1)))
            
            all_dec_mean.append(dec_t)

        return torch.stack(all_dec_mean).permute(1, 0, 2), torch.mean(torch.stack(kld_loss))
    

    def encode(self, y: Tensor,  varBitrate: Tensor, h: Tensor)->Tuple[Tensor, Tensor]:
        '''
        Just do the encoding part of the forward pass with greedy sampling

        y: input tensor, shape (batch, frames, x_dim)
        varBitrate: tensor of the variable bitrate in bits per frame, shape (batch, frames), will be ignored if variableBit is False
        h: initial hidden state, shape (1, batch, h_dim)

        returns: latent tensor (shape (batch, frames, z_dim)), hidden state tensor (shape (1, batch, h_dim))
        '''
        y = (y - self.mean_mel[None, None, :]) / self.std_mel[None, None, :]

        all_z = []
        all_h = []

        phi_x = self.phi_x(y)

        if self.varBit:
            bit_cond_helper = torch.arange(0, self.z_dim, 1).to(y.device)
            bit_mask = varBitrate[:,:,None] > bit_cond_helper[None,None,:]
        else:
            bit_mask = torch.zeros((0,0,0), device=y.device)

        for t in range(y.size(1)):
            phi_x_t = phi_x[:, t, :]

            enc_t = self.enc(torch.cat([phi_x_t, h[-1, :, :]], 1))

            z_t = torch.round(enc_t)

            if self.varBit:
                z_t = z_t * bit_mask[:,t,:].float() + 0.5 * (1-bit_mask[:,t,:].float())

            all_z.append(z_t)
                
            phi_z_t = self.phi_z(z_t)
            
            
            
            dec_t = self.dec(torch.cat([phi_z_t, h[-1, :, :]], 1))
            
            phi_x_t_gen = self.phi_x((dec_t - self.mean_mel[None, :]) / self.std_mel[None, :])
            all_h.append(h[0, :, :])
            _, h = self.rnn(torch.cat([phi_x_t_gen, phi_z_t], 1).unsqueeze(1), h)
            

        return torch.stack(all_z).permute(1, 0, 2), torch.stack(all_h).permute(1, 0, 2)
    
    def decode(self, z: Tensor, h: Tensor)->Tuple[Tensor, Tensor]:
        '''
        Just do the decoding part of the forward pass

        z: latent tensor, shape (batch, frames, z_dim)
        h: initial hidden state, shape (1, batch, h_dim)

        returns: reconstructed tensor (shape (batch, frames, x_dim)), hidden state tensor (shape (1, batch, h_dim))
        '''
        all_dec = []

        for t in range(z.size(1)):
            phi_z_t = self.phi_z(z[:, t, :])
            dec_t = self.dec(torch.cat([phi_z_t, h[-1, :, :]], 1))
            all_dec.append(dec_t)
            phi_x_t_gen = self.phi_x((dec_t - self.mean_mel[None, :]) / self.std_mel[None, :])
            _, h = self.rnn(torch.cat([phi_x_t_gen, phi_z_t], 1).unsqueeze(1), h)

        return torch.stack(all_dec).permute(1, 0, 2), h
