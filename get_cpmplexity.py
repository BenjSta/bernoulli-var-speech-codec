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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_id = "facebook/encodec_24khz"
vocoder_config = toml.load("pretrained_vocoder/config.toml")
vocoder_chkpt_path = "pretrained_vocoder/g_checkpoint"
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




np.random.seed(1)
test_test_examples = np.random.choice(len(clean_test), 5, replace=False)
np.random.seed()

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



import matplotlib.pyplot as plt
np.random.seed(1)
for idx,(y,) in enumerate(tqdm.tqdm(test_dataloader)):
    if idx in test_test_examples:
        with torch.no_grad():
            y_resampled = scipy.signal.resample_poly(y.cpu().numpy()[0, :], vocoder_config['fs'], 48000)
            y_mel = mel_spectrogram(torch.from_numpy(y_resampled).to(device)[None, :], **mel_spec_config)
            plt.close('all')
            plt.figure()
            plt.subplot(3, 1, 1)
            plt.pcolor(y_mel[0, :, :].cpu().numpy())
            varBit_T = 16 * torch.ones(y_mel.shape[0],y_mel.shape[2]).to(device)
            z_code = bvrnn.encode(y_mel.permute(0, 2, 1), varBit_T)
            plt.subplot(3, 1, 2)
            plt.pcolor(z_code[0, :, :].cpu().numpy().T)
            y_mel_reconst = bvrnn.decode(z_code)
            plt.subplot(3, 1, 3)
            plt.pcolor(y_mel_reconst[0, :, :].cpu().numpy().T)
            plt.savefig('encode_decode%d.png' % idx)
            y_mel_reconst = y_mel_reconst.permute(0, 2, 1)
            y_g_hat = generator(y_mel_reconst, y.shape[1])