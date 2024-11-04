import toml
from third_party.BigVGAN.env import AttrDict
from sklearn.preprocessing import StandardScaler
import platform
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import scipy.signal
import tqdm
import os
from dataset import SpeechDataset
from ptflops import get_model_complexity_info
import torch
from third_party.BigVGAN.meldataset import mel_spectrogram
from collections import OrderedDict

from third_party.BigVGAN.models import (
    BigVGAN
)

from third_party.BigVGAN.utils import load_checkpoint
from bvrnn import BVRNN
from dataset import SpeechDataset, load_paths
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


# LOAD CONFIG
config = toml.load("configs_coding/config_varBitRate.toml")
vocoder_config_attr_dict = AttrDict(config['vocoder_config'])
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

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
clean_train, clean_val, _ = paths["clean"]
_, txt_val, _ = paths["txt"]


np.random.seed(1)
val_tensorboard_examples = np.random.choice(len(clean_val), 15, replace=False)
np.random.seed()


trainset = SpeechDataset(
    clean_train,
    duration=config["train_seq_duration"],
    fs=config["fs"],
)

mel_spec_config = {'n_fft': config["winsize"],
                   'num_mels': config["num_mels"],
                   'sampling_rate': config["fs"],
                   'hop_size': config["hopsize"],
                   'win_size': config["winsize"],
                   'fmin': config["fmin"],
                   'fmax': config["fmax"],
                   'padding_left': config["mel_pad_left"]}

mel_spec_huge = []
np.random.seed(1)
for i in tqdm.tqdm(np.random.choice(len(clean_train), 4500, replace=False)):
    y = trainset.__getitem__(i)[0]
    s_mel_spectrogram = mel_spectrogram(torch.from_numpy(y[None, :]), **mel_spec_config)
    mel_spec_huge.append(s_mel_spectrogram)
np.random.seed()

# perform pca on mel_spectrogram
mel_spec_huge = np.concatenate(mel_spec_huge, -1)[0, :, :].T
num_time_frames = mel_spec_huge.shape[0]
num_supervectors = num_time_frames // 3
valid_length = num_supervectors * 3
mel_spec_huge = mel_spec_huge[:valid_length, :]
mel_spec_huge = mel_spec_huge.reshape(num_supervectors, 3 * mel_spec_huge.shape[1])


mu = np.mean(mel_spec_huge)
mel_spec_huge = mel_spec_huge - mu
u, s, vh = np.linalg.svd(mel_spec_huge, full_matrices=False)
mel_spec_orth = mel_spec_huge @ vh.T



from sklearn.cluster import KMeans
import pickle

N_BITS_PER_SPLIT = 8
N_SPLITS = 24

quantizers = []
#scale_factors = []

residual = mel_spec_orth
for _ in tqdm.tqdm(range(N_SPLITS)):
    # Perform VQ in orthogonal space

    residual_norm = np.mean(np.sum(residual**2, -1))
    #scale_factor = 1 / np.sqrt(residual_norm)
    #residual *= scale_factor

    q = KMeans(n_clusters=2**N_BITS_PER_SPLIT, n_init=1, verbose=True, max_iter=1000)
    q.fit(residual)
    residual -= q.cluster_centers_[q.predict(residual), :]

    #scale_factors.append(scale_factor)
    quantizers.append(q)


print(quantizers)

# save quantizers and scale factors
with open("vq_coeffs/quantizers.pkl", "wb") as f:
    pickle.dump(quantizers, f)

with open("vq_coeffs/vh.pkl", "wb") as f:
    pickle.dump(vh, f)

with open("vq_coeffs/mu.pkl", "wb") as f:
    pickle.dump(mu, f)



    