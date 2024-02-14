#%%
from bvrnn import VRNNBernoulli
from ptflops import get_model_complexity_info
import torch
from speech_dataset import SpeechDataset
from meldataset import mel_spectrogram
from transformers import AutoProcessor, EncodecModel
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,'
import time
import argparse
import json
import torch
import soundfile
import tqdm
import subprocess
import scipy.signal
import glob
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
import numpy as np
from ptflops import get_model_complexity_info
import platform
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import sys
from metrics import compute_mean_dnsmos, compute_mean_wacc

# sys.path.append('../Python-WORLD')
# from world import main
torch.backends.cudnn.benchmark = True
# from dnsmos_local import ComputeScore
# import whisper
# from werpy import wer as compute_wer
#%%
VOCALSET_PATH = 'C:/Users/Benja/Desktop/DNS4/clean_fullband/VocalSet_48kHz_mono'

VCTK_VAL_PATHS = [
    'C:/Users/Benja/Desktop/DNS4/clean_fullband/vctk_wav48_silence_trimmed/p225',
    'C:/Users/Benja/Desktop/DNS4/clean_fullband/vctk_wav48_silence_trimmed/p226',
    'C:/Users/Benja/Desktop/DNS4/clean_fullband/vctk_wav48_silence_trimmed/p227',
    'C:/Users/Benja/Desktop/DNS4/clean_fullband/vctk_wav48_silence_trimmed/p228'
]

VCTK_TEST_PATHS = [
    'C:/Users/Benja/Desktop/DNS4/clean_fullband/vctk_wav48_silence_trimmed/p229',
    'C:/Users/Benja/Desktop/DNS4/clean_fullband/vctk_wav48_silence_trimmed/p230',
    'C:/Users/Benja/Desktop/DNS4/clean_fullband/vctk_wav48_silence_trimmed/p232',
    'C:/Users/Benja/Desktop/DNS4/clean_fullband/vctk_wav48_silence_trimmed/p237'
]

np.random.seed(1)

validation_filelist = []
for p in VCTK_VAL_PATHS:
    validation_filelist += np.random.choice(
        sorted(glob.glob(os.path.join(p, '*.wav'), recursive=True)), 30, replace=False).tolist()
val_tensorboard_examples = np.sort(np.random.choice(len(validation_filelist), 20, replace=False))
np.random.seed()

TXT_ROOT = 'C:/Users/Benja/Desktop/DNS4/vctk_txt'

val_texts = []
for v in validation_filelist:
    (dir, file) = os.path.split(v)
    (_, speakerdir) = os.path.split(dir)
    textfile = os.path.join(TXT_ROOT, speakerdir, file[:-9] + '.txt')
    with open(textfile, 'r') as f:
        text = f.read()
        val_texts.append(text[:-1])


scaler_mel = StandardScaler()
scaler_complex = StandardScaler()
scaler_mel_y = StandardScaler()

device='cuda'

if platform.system() == 'Linux':
    def numpy_random_seed(ind=None):
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    worker_init_fn = numpy_random_seed
else:
    worker_init_fn = None


validation_dataset = SpeechDataset(validation_filelist, 
                                    duration=None,
                                    fs=48000)
val_dataloader = DataLoader(validation_dataset, 1, False, None, None, 0)


np.random.seed(1)
        

y_all = []
lyra_all_32 = []
lyra_all_6 = []
lyra_all_92 = []

def lyra(y, fs, bitrate=6, tmpname='tmp'):
    soundfile.write('%s.wav'%tmpname, scipy.signal.resample_poly(y, 16000, fs), 16000)
    os.chdir('lyra')
    print(os.getcwd())
    time.sleep(1)
    cmd_str = '"bazel-bin/lyra/cli_example/encoder_main.exe" --input_path=./../%s.wav --output_dir=../ --bitrate=%d'%(tmpname, bitrate)
    print(cmd_str)
    print(subprocess.call(cmd_str))
    os.system('rm ../%s.wav'%tmpname)
    print(subprocess.call('"bazel-bin/lyra/cli_example/decoder_main" --encoded_path=./../%s.lyra --output_dir=./../ --bitrate=%d'%(tmpname, bitrate)))
    os.chdir('..')
    y, _ = soundfile.read('%s_decoded.wav'%tmpname)
    os.system('rm %s_decoded.wav'%tmpname)
    y = scipy.signal.resample_poly(y, fs, 16000)
    return y.astype('float32')
    

for batch in tqdm.tqdm(val_dataloader):
    (y,) = batch
    #y = y.to(device)
    #inputs = processor(raw_audio=scipy.signal.resample_poly(
    #    y.detach().cpu().numpy()[0, ...], 24000, 48000), return_tensors="pt", sampling_rate=24000)
    lyra_all_32.append(lyra(y.detach().cpu().numpy()[0, ...], 48000, 3200, 'tmp'))
    lyra_all_6.append(lyra(y.detach().cpu().numpy()[0, ...], 48000, 6000, 'tmp'))
    lyra_all_92.append(lyra(y.detach().cpu().numpy()[0, ...], 48000, 9200, 'tmp'))

#%%
dnsmos_lyra32 = compute_mean_dnsmos(lyra_all_32, 48000)
wacc_lyra32, asr_model = compute_mean_wacc(lyra_all_32, val_texts, 48000, model=None)

dnsmos_lyra6 = compute_mean_dnsmos(lyra_all_6, 48000)
wacc_lyra6 = compute_mean_wacc(lyra_all_6, val_texts, 48000, model=asr_model)

dnsmos_lyra92 = compute_mean_dnsmos(lyra_all_92, 48000)
wacc_lyra92 = compute_mean_wacc(lyra_all_92, val_texts, 48000, model=asr_model)


# %%
sw_lyra32 = SummaryWriter(os.path.join('main_logs2/lyra3.2kbps'))
sw_lyra6 = SummaryWriter(os.path.join('main_logs2/lyra6kbps'))
sw_lyra92 = SummaryWriter(os.path.join('main_logs2/lyra9.2kbps'))

sw_clean = SummaryWriter(os.path.join('main_logs2/clean'))
sw_lyra32.add_scalar('DNSMOS-OVR', dnsmos_lyra32[0], 0)
sw_lyra6.add_scalar('DNSMOS-OVR', dnsmos_lyra6[0], 0)
sw_lyra92.add_scalar('DNSMOS-OVR', dnsmos_lyra92[0], 0)

sw_lyra32.add_scalar('Wacc', wacc_lyra32, 0)
sw_lyra6.add_scalar('Wacc', wacc_lyra6, 0)
sw_lyra92.add_scalar('Wacc', wacc_lyra92, 0)

for i in val_tensorboard_examples:
    sw_lyra32.add_audio('%d'%i, torch.from_numpy(lyra_all_32[i]), sample_rate=48000)
    sw_lyra6.add_audio('%d'%i, torch.from_numpy(lyra_all_6[i]), sample_rate=48000)
    sw_lyra92.add_audio('%d'%i, torch.from_numpy(lyra_all_92[i]), sample_rate=48000)


#%%
