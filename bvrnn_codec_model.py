import torch
from torch import nn
from third_party.BigVGAN.meldataset import mel_spectrogram
from bvrnn import BVRNN
from third_party.BigVGAN.models import BigVGAN
from third_party.BigVGAN.env import AttrDict
import numpy as np
import toml
import os

root = os.path.abspath(os.path.split(__file__)[0])

default_config = os.path.join(root, './configs/config_varBitRate.toml')
default_chkpt_bvrnn = os.path.join(root, './chkpts/bvrnn_var_bitrate_step200000')
default_chkpt_vocoder = os.path.join(root, './chkpts/bigvgan_causal_tiny_ftbvrnn_g_step3500000')

SCALING = 10**(-10/20)

class BVRNNCodecModel(nn.Module):
    def __init__(self, config_path=default_config, bvrnn_chkpt_path=default_chkpt_bvrnn, vocoder_chkpt_path=default_chkpt_vocoder):
        '''
        config_path: path to the toml config file
        bvrnn_chkpt_path: path to the checkpoint of the BVRNN model
        vocoder_chkpt_path: path to the checkpoint of the vocoder model
        '''
        super().__init__()
        conf = toml.load(config_path)
        self.conf = conf

        self.bvrnn = BVRNN(x_dim=80, h_dim=conf['h_dim'],
                           z_dim=conf['z_dim'],
                           mean_std_mel=[np.zeros(80), np.ones(80)],
                           log_sigma_init=conf['log_sigma_init'],
                           variableBit=conf['var_bit'])

        self.vocoder = BigVGAN(AttrDict(conf['vocoder_config']))

        bvrnn_chkpt = torch.load(bvrnn_chkpt_path)
        vocoder_chkpt = torch.load(vocoder_chkpt_path)

        self.bvrnn.load_state_dict(bvrnn_chkpt['vrnn'])
        self.vocoder.load_state_dict(vocoder_chkpt['generator'])

    def encode(self, x, bitrate):
        '''
        x: input waveform, shape (batch, length)
        bitrate: target bitrate in bits per second, will be rounded to the nearest valid bitrate
        '''
        xmel = mel_spectrogram(x * SCALING, n_fft=self.conf['winsize'],
                               num_mels=self.conf['num_mels'],
                               sampling_rate=self.conf['fs'],
                               hop_size=self.conf['hopsize'],
                               win_size=self.conf['winsize'],
                               fmin=self.conf['fmin'],
                               fmax=self.conf['fmax'],
                               padding_left=self.conf['mel_pad_left']).permute(0, 2, 1)

        bits_per_frame = np.round(bitrate * self.conf['hopsize'] / self.conf['fs']) *\
            torch.ones((xmel.shape[0], xmel.shape[1])).to(xmel.device)
        h = torch.zeros(1, xmel.shape[0], self.bvrnn.h_dim).to(x.device)
        codes, _ = self.bvrnn.encode(xmel, bits_per_frame, h)
        return codes

    def decode(self, codes, length):
        '''
        codes: latent binary codes, shape (batch, frames, z_dim)
        length: length of the output waveform
        '''
        h = torch.zeros(1, codes.shape[0], self.bvrnn.h_dim).to(codes.device)
        xmel, _ = self.bvrnn.decode(codes, h)
        return self.vocoder(xmel.permute(0, 2, 1), length).squeeze(1) / SCALING

    def forward(self, x, bitrate):
        length = x.shape[1]
        codes = self.encode(x, bitrate)
        return self.decode(codes, length)
