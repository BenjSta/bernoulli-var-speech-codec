import warnings
from dataset import SpeechDataset, load_paths
import toml
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import scipy.signal
from torch.utils.data import DataLoader
import tqdm
import os
import torch
from metrics import compute_dnsmos, compute_pesq, compute_mean_wacc, compute_nisqa, compute_mcd, compute_estimated_metrics, compute_visqol
import whisper
import soundfile
import time
import tempfile
import subprocess
from transformers import AutoProcessor, EncodecModel
from third_party.BigVGAN.models import BigVGAN
from third_party.BigVGAN.utils import load_checkpoint
from third_party.BigVGAN.env import AttrDict
from third_party.BigVGAN.meldataset import mel_spectrogram
from ptflops import get_model_complexity_info
from bvrnn import BVRNN
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model_id = "facebook/encodec_24khz"
vocoder_config = toml.load("pretrained_vocoder/config.toml")
vocoder_chkpt_path = "pretrained_vocoder/g_checkpoint"
encodec_model = EncodecModel.from_pretrained(model_id)
encodec_processor = AutoProcessor.from_pretrained(model_id)
# config = toml.load("configs_coding/config_32bit.toml")
config = toml.load("configs_coding/config_varBitRate.toml")

try:
    chkpt_log_dirs = toml.load('chkpt_log_dirs.toml')
except:
    raise RuntimeError(
        'Error loading chkpt_log_dirs.toml, please create it based on the corresponding default file')

warnings.simplefilter(action="ignore", category=FutureWarning)

torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    pathconfig = toml.load("data_directories.toml")
except:
    raise RuntimeError(
        'Error loading data_directories.toml, please create it based on the corresponding default file')
try:
    executables = toml.load("executable_paths.toml")
except:
    raise RuntimeError(
        'Error loading executable_paths.toml, please create it based on the corresponding default file')

paths = load_paths(pathconfig["DNS4_root"], pathconfig["VCTK_txt_root"])
clean_train, clean_val, clean_test = paths["clean"]
_, txt_val, txt_test = paths["txt"]




test_test_examples = np.array([283])

# np.random.seed(1)
# test_tensorboard_examples = np.random.choice(5, 5, replace=False)
# np.random.seed()

test_dataset = SpeechDataset(
    clean_test,
    duration=None,
    fs=48000,
)

test_dataloader = DataLoader(
    test_dataset,
    1,
    False,
    None,
    None,
    0,
)


mel_spec_config = {'n_fft': vocoder_config["winsize"],
                   'num_mels': vocoder_config["num_mels"],
                   'sampling_rate': vocoder_config["fs"],
                   'hop_size': vocoder_config["hopsize"],
                   'win_size': vocoder_config["winsize"],
                   'fmin': vocoder_config["fmin"],
                   'fmax': vocoder_config["fmax"],
                   'padding_left': vocoder_config["mel_pad_left"]}


vocoder_config_attr_dict = AttrDict(vocoder_config)


# load a vocoder for waveform generation
generator = BigVGAN(vocoder_config_attr_dict).to(device)
print("Generator params: {}".format(sum(p.numel()
      for p in generator.parameters())))

state_dict_g = load_checkpoint(vocoder_chkpt_path, device)
generator.load_state_dict(state_dict_g["generator"])
generator.eval()
print(generator)

# load model 
bvrnn = BVRNN(config["num_mels"], config["h_dim"], config["z_dim"],
            [np.ones([80,]), np.zeros([80,])], config["log_sigma_init"], config["var_bit"]).to(device)
script_bvrnn = torch.jit.script(bvrnn)

chkpt_dir = os.path.join(chkpt_log_dirs['chkpt_log_dir'], config["train_name"], 'checkpoints')
os.makedirs(chkpt_dir, exist_ok=True)


state_dict_vrnn = load_checkpoint(os.path.join(chkpt_dir, "latest"), device)
bvrnn.load_state_dict(state_dict_vrnn["vrnn"])
steps = state_dict_vrnn["steps"]


def constr(input_res):
    return {
        "x": torch.ones((1, 80, input_res[0])).to(device),
        "length": 1 * vocoder_config["fs"],
    }


macs, params = get_model_complexity_info(
    generator,
    (1 * 82,),
    input_constructor=constr,
    as_strings=False,
    print_per_layer_stat=True,
    verbose=True,
)
print("Computational complexity of vocoder model: %g" % macs)
print("Number of parameters in vocoder model: %g" % params)


import pickle
with open('vq_coeffs/quantizers.pkl', 'rb') as f:
    quantizers = pickle.load(f)
with open('vq_coeffs/mu.pkl', 'rb') as f:
    mu = pickle.load(f)
with open('vq_coeffs/vh.pkl', 'rb') as f:
    vh = pickle.load(f)


def quantizer_encode_decode(mel_spec):
    num_time_frames = mel_spec.shape[0]
    num_supervectors = num_time_frames // 3
    valid_length = num_supervectors * 3
    mel_spec = mel_spec[:valid_length, :]
    mel_spec = mel_spec.reshape(num_supervectors, 3 * mel_spec.shape[1])

    mel_spec = mel_spec - mu
    residual = mel_spec @ vh.T
    
    reconst = np.zeros_like(mel_spec)
    for q in quantizers:
        quantized = q.cluster_centers_[q.predict(residual), :]
        residual = residual - quantized
        reconst += quantized
    
    reconst = reconst @ vh
    reconst = reconst + mu
    reconst = reconst.reshape(valid_length, -1)
    return reconst


import librosa
fvec = librosa.mel_frequencies(n_mels=mel_spec_config['num_mels'], fmin=mel_spec_config['fmin'], fmax=mel_spec_config['fmax'])
fvec = np.arange(80)
import matplotlib.pyplot as plt
np.random.seed(1)
for idx,(y,) in enumerate(tqdm.tqdm(test_dataloader)):
    if idx in test_test_examples:
        with torch.no_grad():
            y_resampled = scipy.signal.resample_poly(y.cpu().numpy()[0, :], vocoder_config['fs'], 48000)
            y_mel = mel_spectrogram(torch.from_numpy(y_resampled).to(device)[None, :], **mel_spec_config)
            y_mel = y_mel[:, :, 60:]

            y_mel_reconst = quantizer_encode_decode(y_mel[0, :, :].cpu().numpy().T)


            t = np.arange(y_mel.shape[2]) * 256/22050

            plt.close('all')
            plt.figure(figsize=(6, 2))
            plt.subplot(2, 1, 1)
            plt.pcolor(t, fvec, 8.685889638 * y_mel[0, :, :].cpu().numpy(), vmin=-80, vmax=-10)
            plt.colorbar(label='power / dB')
            plt.subplot(2, 1, 2)
            plt.pcolor(t[:y_mel_reconst.shape[0]], fvec, 8.685889638 * y_mel_reconst.T, vmin=-80, vmax=-10) 
            plt.colorbar(label='power / dB')
            plt.ylabel('Mel band index')
            plt.xlabel('time / s')
            plt.savefig('encode_decode_vq32_%d.png' % idx)

            y_mel_reconst = torch.from_numpy(y_mel_reconst[None, :, :].astype('float32')).permute(0, 2, 1).to(device)
            y_g_hat = generator(y_mel_reconst, y.shape[1])
            y_g_hat = y_g_hat.squeeze().cpu().numpy()
            soundfile.write('encode_decode_vq32_%d.wav' % idx, y_g_hat, 22050)

            f, axs = plt.subplots(10, 2, figsize=(6, 14), 
                                  gridspec_kw={'width_ratios': [10, 0.3],
                                               'height_ratios': [48,48, 8, 48, 48, 16, 48, 48, 48, 48],
                                               'wspace': 0.05, 'hspace': 0.1})
            
            
            
            #plt.tick_params(bottom=False, labelbottom=False)
            h = axs[0][0].pcolor(t, fvec, 8.685889638 * y_mel[0, :, :].cpu().numpy(), vmin=-80, vmax=-10)
        
            plt.colorbar(h, label='power / dB', cax=axs[0][1])
            axs[0][0].set_ylabel('Mel band index')
            axs[0][0].set_xlabel('time / s')
            axs[1][0].set_visible(False)
            axs[4][0].set_visible(False)
            axs[7][0].set_visible(False)
            axs[1][1].set_visible(False)
            
            axs[4][1].set_visible(False)
            axs[7][1].set_visible(False)
            axs[2][1].set_visible(False)
            axs[5][1].set_visible(False)
            axs[8][1].set_visible(False)

            # axs[0][1].set_visible(False)
            # axs[3][1].set_visible(False)
            # axs[6][1].set_visible(False)
            # axs[9][1].set_visible(False)

            for b, bitrate in enumerate([4, 16, 64]):
                varBit_T = bitrate * torch.ones(y_mel.shape[0],y_mel.shape[2]).to(device)
                z_code, h2 = bvrnn.encode(y_mel.permute(0, 2, 1), varBit_T, torch.zeros(1, y_mel.shape[0], bvrnn.h_dim).to(device))
            
                #axs[b*2+1].set_title('%d bits per frame' %)
                axs[b*3+2][0].set_xticks([], [])
                axs[b*3+2][0].set_yticks([1, bitrate])
                #plt.subplots_adjust(hspace=1)
                axs[b*3+2][0].pcolor(t, np.arange(bitrate)+1, z_code[0, :, :bitrate].cpu().numpy().T)
                
                axs[b*3+2][0].set_ylabel('latent\ndim.', labelpad = 10)
                axs[b*3+2][0].invert_yaxis()
                
                y_mel_reconst, h2 = bvrnn.decode(z_code, torch.zeros(1, y_mel.shape[0], bvrnn.h_dim).to(device))
                axs[b*3+3][0].tick_params(bottom=True, labelbottom=True)
                
                h = axs[b*3+3][0].pcolor(t, fvec, 8.685889638 * y_mel_reconst[0, :, :].cpu().numpy().T, vmin=-80, vmax=-10)
                plt.colorbar(h, label='power / dB', cax=axs[b*3+3][1])
                axs[b*3+3][0].set_ylabel('Mel band index')
                axs[b*3+3][0].set_xlabel('time / s')
                y_mel_reconst = y_mel_reconst.permute(0, 2, 1)
                y_g_hat = generator(y_mel_reconst, y.shape[1])
                y_g_hat = y_g_hat.squeeze().cpu().numpy()
                soundfile.write('encode_decode_%d_%d.wav' % (bitrate, idx), y_g_hat, 22050)
            
            
            
            #plt.tight_layout()
            
            plt.savefig('encode_decode%d.png' % idx)
            