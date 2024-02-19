#%%
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
model_id = "facebook/encodec_24khz"
model = EncodecModel.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

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
encodec_all_15 = []
encodec_all_30 = []
encodec_all_60 = []
encodec_all_120 = []
    
for batch in tqdm.tqdm(val_dataloader):
    (y,) = batch
    #y = y.to(device)
    inputs = processor(raw_audio=scipy.signal.resample_poly(
        y.detach().cpu().numpy()[0, ...], 24000, 48000), return_tensors="pt", sampling_rate=24000)

    
    outputs = model(**inputs, bandwidth=1.5)
    audio_codes = outputs.audio_codes
    audio_values = outputs.audio_values
    encodec_all_15.append(scipy.signal.resample_poly(
        audio_values.detach().cpu().numpy()[0,0,:], 48000, 24000))
    
    outputs = model(**inputs, bandwidth=3.0)
    audio_codes = outputs.audio_codes
    audio_values = outputs.audio_values
    encodec_all_30.append(scipy.signal.resample_poly(
        audio_values.detach().cpu().numpy()[0,0,:], 48000, 24000))
    
    outputs = model(**inputs, bandwidth=6.0)
    audio_codes = outputs.audio_codes
    audio_values = outputs.audio_values
    encodec_all_60.append(scipy.signal.resample_poly(
        audio_values.detach().cpu().numpy()[0,0,:], 48000, 24000))
    
    outputs = model(**inputs, bandwidth=12)
    audio_codes = outputs.audio_codes
    audio_values = outputs.audio_values
    encodec_all_120.append(scipy.signal.resample_poly(
        audio_values.detach().cpu().numpy()[0,0,:], 48000, 24000))
    

    #loss_all.append(mae.detach().cpu().numpy())
    #lengths_all.append(y_hat_reconst.shape[1])
    y_all.append(y.detach().cpu().numpy()[0, :])
    

#%%

dnsmos_encodec15 = compute_mean_dnsmos(encodec_all_15, 48000)
wacc_encodec15, asr_model = compute_mean_wacc(encodec_all_15, val_texts, 48000, model=None)

dnsmos_encodec30 = compute_mean_dnsmos(encodec_all_30, 48000)
wacc_encodec30 = compute_mean_wacc(encodec_all_30, val_texts, 48000, model=asr_model)
#%%
dnsmos_encodec60 = compute_mean_dnsmos(encodec_all_60, 48000)
wacc_encodec60 = compute_mean_wacc(encodec_all_60, val_texts, 48000, model=asr_model)

dnsmos_encodec120 = compute_mean_dnsmos(encodec_all_120, 48000)
wacc_encodec120 = compute_mean_wacc(encodec_all_120, val_texts, 48000, model=asr_model)

dnsmos_clean = compute_mean_dnsmos(y_all, 48000)
wacc_clean = compute_mean_wacc(y_all, val_texts, 48000, model=asr_model)

# %%
sw_encodec15 = SummaryWriter(os.path.join('main_logs2/encodec1.5kbps'))
sw_encodec30 = SummaryWriter(os.path.join('main_logs2/encodec3.0kbps'))
sw_encodec60 = SummaryWriter(os.path.join('main_logs2/encodec6.0kbps'))
sw_encodec120 = SummaryWriter(os.path.join('main_logs2/encodec12.0kbps'))

sw_clean = SummaryWriter(os.path.join('main_logs2/clean'))
sw_encodec15.add_scalar('DNSMOS-OVR', dnsmos_encodec15[0], 0)
sw_encodec30.add_scalar('DNSMOS-OVR', dnsmos_encodec30[0], 0)
sw_encodec60.add_scalar('DNSMOS-OVR', dnsmos_encodec60[0], 0)
sw_encodec120.add_scalar('DNSMOS-OVR', dnsmos_encodec120[0], 0)
sw_clean.add_scalar('DNSMOS-OVR', dnsmos_clean[0], 0)

sw_encodec15.add_scalar('Wacc', wacc_encodec15, 0)
sw_encodec30.add_scalar('Wacc', wacc_encodec30, 0)
sw_encodec60.add_scalar('Wacc', wacc_encodec60, 0)
sw_encodec120.add_scalar('Wacc', wacc_encodec120, 0)
sw_clean.add_scalar('Wacc', wacc_clean, 0)

for i in val_tensorboard_examples:
    sw_encodec15.add_audio('%d'%i, torch.from_numpy(encodec_all_15[i]), sample_rate=48000)
    sw_encodec30.add_audio('%d'%i, torch.from_numpy(encodec_all_30[i]), sample_rate=48000)
    sw_encodec60.add_audio('%d'%i, torch.from_numpy(encodec_all_60[i]), sample_rate=48000)
    sw_encodec120.add_audio('%d'%i, torch.from_numpy(encodec_all_120[i]), sample_rate=48000)

    sw_clean.add_audio('%d'%i, torch.from_numpy(y_all[i]), sample_rate=48000)

#%%
