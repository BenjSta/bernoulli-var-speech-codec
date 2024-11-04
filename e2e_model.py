import torch
from torch import nn
from third_party.BigVGAN.meldataset import mel_spectrogram
from bvrnn import BVRNN
from third_party.BigVGAN.models import BigVGAN
from third_party.BigVGAN.env import AttrDict

class E2EModel(nn.Module):
    def __init__(self, bvrnn_config, vocoder_config):
        super(E2EModel).__init__()




        self.bvrnn = BVRNN(**bvrnn_config)
        self.vocoder = BigVGAN(AttrDict(vocoder_config))
    
    def encode(self, x, bits_per_frame):
        xmel = mel_spectrogram(x)
        h = torch.zeros(1, xmel.shape[0], self.bvrnn.h_dim).to(x.device)
        codes = self.bvrnn.encode(xmel, bits_per_frame, h)
        return codes
    
    def decode(self, codes, bits_per_frame):
        xmel = self.bvrnn.decode()

    def forward(self, x, bits_per_frame):
        xmel = mel_spectrogram(x)






