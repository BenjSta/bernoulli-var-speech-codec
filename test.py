import pickle
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
MAKEUP_GAIN = 10.0


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_id = "facebook/encodec_24khz"
vocoder_config = toml.load(
    "pretrained_and_finetuned_vocoder_chkpts/config_bigvgan_causal_tiny.toml")
vocoder_config_attr_dict = AttrDict(vocoder_config)

vocoder_chkpt_path = "pretrained_and_finetuned_vocoder_chkpts/bigvgan_causal_tiny_g_2500000"
vocoder_chkpt_path_ft = "pretrained_and_finetuned_vocoder_chkpts/bigvgan_causal_tiny_ftbvrnn_g_2860000"
vocoder_chkpt_path_ftvq = "pretrained_and_finetuned_vocoder_chkpts/bigvgan_causal_tiny_ftvq_g_3380000"


vocoder_config_bigger = toml.load(
    "configs_vocoder_training/config_vocoder_causal_snake_doubleseg_bigger.toml")
vocoder_config_bigger_attr_dict = AttrDict(vocoder_config_bigger)
vocoder_chkpt_path_bigger = "/media/DATA/shared/stahl/bernoulli-var-speech-codec/vocoder/checkpoints/snake_causal_doubleseg_bigger/g_2500000"
vocoder_config_bigger_sym = toml.load(
    "pretrained_and_finetuned_vocoder_chkpts/config_bigvgan_base.toml")
vocoder_chkpt_path_bigger_sym = "/media/DATA/shared/stahl/bernoulli-var-speech-codec/vocoder/checkpoints/snake_noncausal_bigger/g_2500000"
vocoder_config_bigger_sym_attr_dict = AttrDict(vocoder_config_bigger_sym)

encodec_model = EncodecModel.from_pretrained(model_id)
encodec_processor = AutoProcessor.from_pretrained(model_id)

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


with open('vq_coeffs/quantizers2.pkl', 'rb') as f:
    quantizers = pickle.load(f)
with open('vq_coeffs/mu.pkl', 'rb') as f:
    mu = pickle.load(f)
with open('vq_coeffs/vh.pkl', 'rb') as f:
    vh = pickle.load(f)


def quantizer_encode_decode(mel_spec, layers):
    num_time_frames = mel_spec.shape[0]
    num_supervectors = num_time_frames // 3
    valid_length = num_supervectors * 3
    mel_spec = mel_spec[:valid_length, :]
    mel_spec = mel_spec.reshape(num_supervectors, 3 * mel_spec.shape[1])

    mel_spec = mel_spec - mu
    residual = mel_spec @ vh.T

    reconst = np.zeros_like(mel_spec)

    for q in quantizers[:layers]:
        quantized = q.cluster_centers_[q.predict(residual), :]
        residual = residual - quantized
        reconst += quantized

    reconst = reconst @ vh
    reconst = reconst + mu
    reconst = reconst.reshape(valid_length, -1)
    return reconst


np.random.seed(1)
test_test_examples = np.random.choice(len(clean_test), 20, replace=False)
np.random.seed()

np.random.seed(1)
test_tensorboard_examples = np.random.choice(20, 5, replace=False)
np.random.seed()


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
mel_spec_config_sym = {'n_fft': vocoder_config_bigger_sym["winsize"],
                    'num_mels': vocoder_config_bigger_sym["num_mels"],
                    'sampling_rate': vocoder_config_bigger_sym["fs"],
                    'hop_size': vocoder_config_bigger_sym["hopsize"],
                    'win_size': vocoder_config_bigger_sym["winsize"],
                    'fmin': vocoder_config_bigger_sym["fmin"],
                    'fmax': vocoder_config_bigger_sym["fmax"],
                    'padding_left': vocoder_config_bigger_sym["mel_pad_left"]}



# load 5 different vocoders
generator = BigVGAN(vocoder_config_attr_dict).to(device)
state_dict_g = load_checkpoint(vocoder_chkpt_path, device)
generator.load_state_dict(state_dict_g["generator"])
generator.eval()

generator_ft = BigVGAN(vocoder_config_attr_dict).to(device)
state_dict_g = load_checkpoint(vocoder_chkpt_path_ft, device)
generator_ft.load_state_dict(state_dict_g["generator"])
generator_ft.eval()

generator_ftvq = BigVGAN(vocoder_config_attr_dict).to(device)
state_dict_g = load_checkpoint(vocoder_chkpt_path_ft, device)
generator_ftvq.load_state_dict(state_dict_g["generator"])
generator_ftvq.eval()

generator_big = BigVGAN(vocoder_config_bigger_attr_dict).to(device)
state_dict_g = load_checkpoint(vocoder_chkpt_path_bigger, device)
generator_big.load_state_dict(state_dict_g["generator"])
generator_big.eval()

generator_bigsym = BigVGAN(vocoder_config_bigger_sym_attr_dict).to(device)
state_dict_g = load_checkpoint(vocoder_chkpt_path_bigger_sym, device)
generator_bigsym.load_state_dict(state_dict_g["generator"])
generator_bigsym.eval()


# load model
# variable bitrate
bvrnn_var = BVRNN(config["num_mels"], config["h_dim"], config["z_dim"],
                  [np.ones([80,]), np.zeros([80,])], config["log_sigma_init"], config["var_bit"]).to(device)
script_bvrnn_var = torch.jit.script(bvrnn_var)
chkpt_dir = os.path.join(
    chkpt_log_dirs['chkpt_log_dir'], "variable_BitrateKLMask", 'checkpoints')
state_dict_vrnn = load_checkpoint(os.path.join(chkpt_dir, "latest"), device)
bvrnn_var.load_state_dict(state_dict_vrnn["vrnn"])
# steps = state_dict_vrnn["steps"]
# fixed 16
bvrnn_fix_16 = BVRNN(config["num_mels"], config["h_dim"], 16,
                     [np.ones([80,]), np.zeros([80,])], config["log_sigma_init"], False).to(device)
script_bvrnn_16 = torch.jit.script(bvrnn_fix_16)
chkpt_dir = os.path.join(
    chkpt_log_dirs['chkpt_log_dir'], '16bit_fixed', 'checkpoints')
state_dict_vrnn = load_checkpoint(os.path.join(chkpt_dir, "latest"), device)
bvrnn_fix_16.load_state_dict(state_dict_vrnn["vrnn"])
# fixed 24
bvrnn_fix_24 = BVRNN(config["num_mels"], config["h_dim"], 24,
                     [np.ones([80,]), np.zeros([80,])], config["log_sigma_init"], False).to(device)
script_bvrnn_24 = torch.jit.script(bvrnn_fix_24)
chkpt_dir = os.path.join(
    chkpt_log_dirs['chkpt_log_dir'], '24bit_fixed', 'checkpoints')
state_dict_vrnn = load_checkpoint(os.path.join(chkpt_dir, "latest"), device)
bvrnn_fix_24.load_state_dict(state_dict_vrnn["vrnn"])
# fixed 32
bvrnn_fix_32 = BVRNN(config["num_mels"], config["h_dim"], 32,
                     [np.ones([80,]), np.zeros([80,])], config["log_sigma_init"], False).to(device)
script_bvrnn_32 = torch.jit.script(bvrnn_fix_32)
chkpt_dir = os.path.join(
    chkpt_log_dirs['chkpt_log_dir'], '32bit_fixed', 'checkpoints')
state_dict_vrnn = load_checkpoint(os.path.join(chkpt_dir, "latest"), device)
bvrnn_fix_32.load_state_dict(state_dict_vrnn["vrnn"])
# fixed 64
bvrnn_fix_64 = BVRNN(config["num_mels"], config["h_dim"], 64,
                     [np.ones([80,]), np.zeros([80,])], config["log_sigma_init"], False).to(device)
script_bvrnn_64 = torch.jit.script(bvrnn_fix_64)
chkpt_dir = os.path.join(
    chkpt_log_dirs['chkpt_log_dir'], '64bit_fixed', 'checkpoints')
state_dict_vrnn = load_checkpoint(os.path.join(chkpt_dir, "latest"), device)
bvrnn_fix_64.load_state_dict(state_dict_vrnn["vrnn"])


def constr(input_res):
    return {
        "x": torch.ones((1, 80, input_res[0])).to(device),
        "length": 1 * vocoder_config["fs"],
    }


macs, params = get_model_complexity_info(
    generator,
    (1 * 86,),
    input_constructor=constr,
    as_strings=False,
    print_per_layer_stat=True,
    verbose=True,
)
print("Computational complexity of vocoder model: %g" % macs)
print("Number of parameters in vocoder model: %g" % params)


def opus(y, fs, bitrate=6):
    with tempfile.NamedTemporaryFile(suffix='.wav', mode='r+') as tmpfile1, tempfile.NamedTemporaryFile(suffix='.wav', mode='r+') as tmpfile3, tempfile.NamedTemporaryFile(suffix='.opus', mode='r+') as tmpfile2:
        soundfile.write('%s' % tmpfile1.name, y, fs)
        subprocess.call('/home/stahl/miniconda3/envs/spear/bin/ffmpeg -y -i %s  -c:a libopus -b:a %dK %s' %
                        (tmpfile1.name, bitrate,
                         tmpfile2.name), stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT, shell=True)
        subprocess.call('/home/stahl/miniconda3/envs/spear/bin/ffmpeg -y -i %s -ar %d %s' %
                        (tmpfile2.name, fs, tmpfile3.name), stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT, shell=True)
        y, _ = soundfile.read('%s' % tmpfile3.name)
    return y.astype('float32')


def lyra(y, fs, bitrate=6):
    with tempfile.NamedTemporaryFile(suffix='.wav', mode='r+') as tmpfile, tempfile.TemporaryDirectory() as tmpdir:
        soundfile.write(
            tmpfile.name, scipy.signal.resample_poly(y, 48000, fs), 48000)
        cwd = os.getcwd()
        os.chdir(executables['lyra_base'])
        subprocess.call('%s --input_path="%s" --output_dir="%s" --bitrate=%g' % (
            executables['lyra_encoder'], tmpfile.name, tmpdir, bitrate), shell=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT)

        subprocess.call('%s --encoded_path="%s/%s.lyra" --output_dir="%s" --bitrate=%g' %
                        (executables['lyra_decoder'], tmpdir, os.path.split(tmpfile.name)[-1][:-len('.wav')], tmpdir, bitrate), shell=True, stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT)
        y, fs_decoded = soundfile.read(os.path.join(tmpdir,
                                                    os.path.split(tmpfile.name)[-1][:-len('.wav')]+'_decoded.wav'))
        y = scipy.signal.resample_poly(y, fs, fs_decoded)
        os.chdir(cwd)
    return y.astype('float32')


def encodec(y, fs, bandwidth):
    inputs = encodec_processor(raw_audio=scipy.signal.resample_poly(
        y, 24000, fs), return_tensors="pt", sampling_rate=24000)
    outputs = encodec_model(**inputs, bandwidth=bandwidth)
    audio_values = outputs.audio_values
    return scipy.signal.resample_poly(
        audio_values.detach().cpu().numpy()[0, 0, :], fs, 24000)


clean_all = []

vocoded_all = []
vocoded_big_all = []
vocoded_bigsym_all = []

opus6_all = []
opus8_all = []
opus10_all = []
opus12_all = []

lyra3_2_all = []
lyra6_all = []
lyra9_2_all = []

encodec1_5_all = []
encodec3_all = []
encodec6_all = []
encodec12_all = []

quantizer_all = []
quantizer_ft_all = []

varBit_16_all = []
varBit_24_all = []
varBit_32_all = []
varBit_64_all = []

varBit_16_ft_all = []
varBit_24_ft_all = []
varBit_32_ft_all = []
varBit_64_ft_all = []

varBit_16_big_all = []
varBit_24_big_all = []
varBit_32_big_all = []
varBit_64_big_all = []

varBit_16_bigsym_all = []
varBit_24_bigsym_all = []
varBit_32_bigsym_all = []
varBit_64_bigsym_all = []

fixedBit_16_all = []
fixedBit_24_all = []
fixedBit_32_all = []
fixedBit_64_all = []

fixedBit_16_ft_all = []
fixedBit_24_ft_all = []
fixedBit_32_ft_all = []
fixedBit_64_ft_all = []

quant_16_all = []
quant_24_all = []
quant_32_all = []
quant_64_all = []

quant_16_ft_all = []
quant_24_ft_all = []
quant_32_ft_all = []
quant_64_ft_all = []

L1_fix_16 = []
L1_fix_24 = []
L1_fix_32 = []
L1_fix_64 = []
L1_var_16 = []
L1_var_24 = []
L1_var_32 = []
L1_var_64 = []
L1_quant_16 = []
L1_quant_24 = []
L1_quant_32 = []
L1_quant_64 = []

L2_fix_16 = []
L2_fix_24 = []
L2_fix_32 = []
L2_fix_64 = []
L2_var_16 = []
L2_var_24 = []
L2_var_32 = []
L2_var_64 = []
L2_quant_16 = []
L2_quant_24 = []
L2_quant_32 = []
L2_quant_64 = []

# sws = []

sws = ['clean',
       'vocoder', 'vocoder_big', 'vocoder_big_sym',
       'opus6', 'opus8', 'opus10', 'opus12', 'lyra3_2', 'lyra6',
       'lyra9_2', 'encodec1_5', 'encodec3', 'encodec6', 'encodec12',
       'fixed16', 'fixed24',
       'fixed32', 'fixed64',
       'fixed16_ft', 'fixed24_ft',
       'fixed32_ft', 'fixed64_ft',
       'variable16', 'variable24',
       'variable32', 'variable64',
       'variable16_ft', 'variable24_ft',
       'variable32_ft', 'variable64_ft',
       'variable16_big', 'variable24_big',
       'variable32_big', 'variable64_big',
       'variable16_big_sym', 'variable24_big_sym',
       'variable32_big_sym', 'variable64_big_sym',
       'quantizer16', 'quantizer24', 'quantizer32', 'quantizer64',
       'quantizer16_ft', 'quantizer24_ft', 'quantizer32_ft', 'quantizer64_ft']

sigs_processed = [clean_all, vocoded_all, vocoded_big_all, vocoded_bigsym_all,
        opus6_all, opus8_all, opus10_all, opus12_all, lyra3_2_all, lyra6_all,
        lyra9_2_all, encodec1_5_all, encodec3_all, encodec6_all, encodec12_all,
        fixedBit_16_all, fixedBit_24_all, fixedBit_32_all, fixedBit_64_all,
        fixedBit_16_ft_all, fixedBit_24_ft_all, fixedBit_32_ft_all, fixedBit_64_ft_all,
        varBit_16_all, varBit_24_all, varBit_32_all, varBit_64_all,
        varBit_16_ft_all, varBit_24_ft_all, varBit_32_ft_all, varBit_64_ft_all,
        varBit_16_big_all, varBit_24_big_all, varBit_32_big_all, varBit_64_big_all,
        varBit_16_bigsym_all, varBit_24_bigsym_all, varBit_32_bigsym_all, varBit_64_bigsym_all,
        quant_16_all, quant_24_all, quant_32_all, quant_64_all,
        quant_16_ft_all, quant_24_ft_all, quant_32_ft_all, quant_64_ft_all]

# bitrates =
wavPath = 'WAV/'

distances_all = []
# generator.eval()
np.random.seed(1)
for idx, (y,) in enumerate(tqdm.tqdm(test_dataloader)):
    if idx in test_test_examples:
        with torch.no_grad():
            clean_all.append(y[0, :].numpy())
            y_resampled = scipy.signal.resample_poly(
                y.cpu().numpy()[0, :], vocoder_config['fs'], 48000)
            # y_resampled = y_resaampled * 0.95 / np.max(np.abs(y_resampled))
            y_mel = mel_spectrogram(torch.from_numpy(y_resampled).to(device)[
                                    None, :], **mel_spec_config)
            y_mel_sym = mel_spectrogram(torch.from_numpy(y_resampled).to(device)[
                                        None, :], **mel_spec_config_sym)

            y_g_hat = generator(y_mel, y_resampled.shape[0])
            vocoded_all.append(scipy.signal.resample_poly(y_g_hat[0, 0, :].detach().cpu().numpy(), 48000, config['fs']))
            y_g_hat_big = generator_big(y_mel, y_resampled.shape[0])
            vocoded_big_all.append(scipy.signal.resample_poly(y_g_hat_big[0, 0, :].detach().cpu().numpy(), 48000, config['fs']))
            y_g_hat_bigsym = generator_bigsym(y_mel_sym, y_resampled.shape[0])
            vocoded_bigsym_all.append(scipy.signal.resample_poly(y_g_hat_bigsym[0, 0, :].detach().cpu().numpy(), 48000, config['fs']))

            # variable bitrate
            for varBit, sig, sigft, sigbig, sigbigsym, L1, L2 in zip(
                    [16, 24, 32, 64],
                    [varBit_16_all, varBit_24_all, varBit_32_all, varBit_64_all],
                    [varBit_16_ft_all, varBit_24_ft_all, varBit_32_ft_all, varBit_64_ft_all],
                    [varBit_16_big_all, varBit_24_big_all, varBit_32_big_all, varBit_64_big_all],
                    [varBit_16_bigsym_all, varBit_24_bigsym_all, varBit_32_bigsym_all, varBit_64_bigsym_all],
                    [L1_var_16, L1_var_24, L1_var_32, L1_var_64],
                    [L2_var_16, L2_var_24, L2_var_32, L2_var_64]):
                varBit_T = varBit * \
                    torch.ones(y_mel.shape[0], y_mel.shape[2]).to(device)
                
                y_mel_reconst, kld = bvrnn_var(
                    y_mel.permute(0, 2, 1), 1.0, True, varBit_T)
                y_mel_reconst = y_mel_reconst.permute(0, 2, 1)

                y_mel_reconst_sym, kld = bvrnn_var(
                    y_mel_sym.permute(0, 2, 1), 1.0, True, varBit_T)
                y_mel_reconst_sym = y_mel_reconst_sym.permute(0, 2, 1)

                y_g_hat = generator(y_mel_reconst, y_resampled.shape[0])
                y_g_hat_ft = generator_ft(y_mel_reconst, y_resampled.shape[0])
                y_g_hat_big = generator_big(y_mel_reconst, y_resampled.shape[0])
                y_g_hat_bigsym = generator_bigsym(y_mel_reconst_sym, y_resampled.shape[0])

                sig.append(scipy.signal.resample_poly(
                    y_g_hat[0, 0, :].detach().cpu().numpy(), 48000, config['fs']))
                sigft.append(scipy.signal.resample_poly(
                    y_g_hat_ft[0, 0, :].detach().cpu().numpy(), 48000, config['fs']))
                sigbig.append(scipy.signal.resample_poly(
                    y_g_hat_big[0, 0, :].detach().cpu().numpy(), 48000, config['fs']))
                sigbigsym.append(scipy.signal.resample_poly(
                    y_g_hat_bigsym[0, 0, :].detach().cpu().numpy(), 48000, config['fs']))
                
                L1.append(torch.mean(torch.abs(y_mel[0, :, :] - y_mel_reconst[0, :, :])).detach().cpu().numpy() * 20 / np.log(10))
                L2.append(torch.sqrt(torch.mean(torch.pow(y_mel[0, :, :] - y_mel_reconst[0, :, :],2))).detach().cpu().numpy() * 20 / np.log(10))
                
            for layers, sigs, sigs_ft, L1, L2 in zip([6, 9, 12, 24],[quant_16_all,quant_24_all,quant_32_all,quant_64_all],
                                             [quant_16_ft_all,quant_24_ft_all,quant_32_ft_all,quant_64_ft_all],
                                             [L1_quant_16, L1_quant_24, L1_quant_32, L1_quant_64],
                                             [L2_quant_16, L2_quant_24, L2_quant_32, L2_quant_64],):
                y_mel_reconst = quantizer_encode_decode(y_mel[0, :, :].cpu().numpy().T, layers)
                y_mel_reconst = torch.from_numpy(y_mel_reconst[None, :, :].astype('float32')).permute(0, 2, 1).to(device)
                L1.append(torch.mean(torch.abs(y_mel[0, :, :y_mel_reconst.shape[2]] - y_mel_reconst[0, :, :])).detach().cpu().numpy() * 20 / np.log(10))
                L2.append(torch.sqrt(torch.mean(torch.pow(y_mel[0, :, :y_mel_reconst.shape[2]] - y_mel_reconst[0, :, :],2))).detach().cpu().numpy() * 20 / np.log(10))
                y_g_hat = generator(y_mel_reconst, y_resampled.shape[0])
                sigs.append(scipy.signal.resample_poly(y_g_hat[0, 0, :].detach().cpu().numpy(), 48000, config['fs']))
                y_g_hat_ft = generator_ftvq(y_mel_reconst, y_resampled.shape[0])
                sigs_ft.append(scipy.signal.resample_poly(y_g_hat_ft[0, 0, :].detach().cpu().numpy(), 48000, config['fs']))
  
            #fixed bitrate
            for zdim, model, L1, L2, sig, sig_ft in zip([16,24,32,64], [bvrnn_fix_16, bvrnn_fix_24, bvrnn_fix_32, bvrnn_fix_64],
                                          [L1_fix_16, L1_fix_24, L1_fix_32, L1_fix_64],
                                          [L2_fix_16, L2_fix_24, L2_fix_32, L2_fix_64],
                                          [fixedBit_16_all, fixedBit_24_all, fixedBit_32_all, fixedBit_64_all],
                                          [fixedBit_16_ft_all, fixedBit_24_ft_all, fixedBit_32_ft_all, fixedBit_64_ft_all]):
                varBit = zdim
                varBit_T = varBit * torch.ones(y_mel.shape[0],y_mel.shape[2]).to(device)
                y_mel_reconst, kld = model(y_mel.permute(0, 2, 1), 1.0, True, varBit_T)
                y_mel_reconst = y_mel_reconst.permute(0, 2, 1)
                y_g_hat = generator(y_mel_reconst, y_resampled.shape[0])
                sig.append(scipy.signal.resample_poly(y_g_hat[0, 0, :].detach().cpu().numpy(), 48000, config['fs']))

                y_g_hat = generator_ft(y_mel_reconst, y_resampled.shape[0])
                sig_ft.append(scipy.signal.resample_poly(y_g_hat[0, 0, :].detach().cpu().numpy(), 48000, config['fs']))

                L1.append(torch.mean(torch.abs(y_mel[0, :, :] - y_mel_reconst[0, :, :])).detach().cpu().numpy() * 20 / np.log(10))
                L2.append(torch.sqrt(torch.mean(torch.pow(y_mel[0, :, :] - y_mel_reconst[0, :, :],2))).detach().cpu().numpy() * 20 / np.log(10))

            opus6_all.append(10**(-MAKEUP_GAIN/20)*opus(10**(MAKEUP_GAIN/20)* y[0, :].numpy(), 48000, 6))
            opus8_all.append(10**(-MAKEUP_GAIN/20)*opus(10**(MAKEUP_GAIN/20)* y[0, :].numpy(), 48000, 8))
            opus10_all.append(10**(-MAKEUP_GAIN/20)*opus(10**(MAKEUP_GAIN/20)* y[0, :].numpy(), 48000, 10))
            opus12_all.append(10**(-MAKEUP_GAIN/20)*opus(10**(MAKEUP_GAIN/20)* y[0, :].numpy(), 48000, 12))

            lyra3_2_all.append(10**(-MAKEUP_GAIN/20)*lyra(10**(MAKEUP_GAIN/20)* y[0, :].numpy(), 48000, 3200))
            lyra6_all.append(10**(-MAKEUP_GAIN/20)*lyra(10**(MAKEUP_GAIN/20) * y[0, :].numpy(), 48000, 6000))
            lyra9_2_all.append(10**(-MAKEUP_GAIN/20)*lyra(10**(MAKEUP_GAIN/20)* y[0, :].numpy(), 48000, 9200))

            encodec1_5_all.append(10**(-MAKEUP_GAIN/20)*encodec(10**(MAKEUP_GAIN/20)*y[0, :].numpy(), 48000, 1.5))
            encodec3_all.append(10**(-MAKEUP_GAIN/20)*encodec(10**(MAKEUP_GAIN/20)*y[0, :].numpy(), 48000, 3))
            encodec6_all.append(10**(-MAKEUP_GAIN/20)*encodec(10**(MAKEUP_GAIN/20)*y[0, :].numpy(), 48000, 6))
            encodec12_all.append(10**(-MAKEUP_GAIN/20)*encodec(10**(MAKEUP_GAIN/20)*y[0, :].numpy(), 48000, 12))



lengths_all = np.array([y.shape[0] for y in clean_all])

for sw, sigs_method in zip(sws, sigs_processed):
    df = pd.DataFrame([])
    pesq = compute_pesq(clean_all, sigs_method, 48000)
    # stoi_est, pesq_est, sisdr_est, mos = compute_estimated_metrics(clean_all, sigs_method, 48000, device)
    # ovr, sig, bak = compute_dnsmos(sigs_method, 48000)
    # mcd = compute_mcd(clean_all, sigs_method, 48000)
    visqol = compute_visqol(
        executables['visqol_base'],  executables['visqol_bin'], clean_all, sigs_method, 48000)
    # wacc = compute_mean_wacc(sigs_method, (np.array(txt_test)[sorted(test_test_examples)]).tolist(), 48000, device)
    nisqa16_mos, _, _, _, _, nisqa48_mos, _, _, _, _ = compute_nisqa(
        sigs_method, 48000)
    df.loc[:, 'pesq'] = pesq
    df.loc[:, 'nisqa_mos16'] = nisqa16_mos
    df.loc[:, 'nisqa_mos48'] = nisqa48_mos
    df.loc[:, 'visqol'] = visqol
    # df.loc[:,'wacc'] = wacc
    df.loc[:, 'lengths'] = lengths_all
    fileName = 'testResult/' + sw + '.csv'
    df.to_csv(fileName)


distances = [L1_fix_16, L1_fix_24, L1_fix_32, L1_fix_64, 
             L1_var_16, L1_var_24, L1_var_32, L1_var_64,
             L1_quant_16, L1_quant_24, L1_quant_32, L1_quant_64,
             L2_fix_16, L2_fix_24, L2_fix_32, L2_fix_64,
            L2_var_16, L2_var_24, L2_var_32, L2_var_64,
            L2_quant_16, L2_quant_24, L2_quant_32, L2_quant_64]

names = ['L1_fix_16', 'L1_fix_24', 'L1_fix_32', 'L1_fix_64',
         'L1_var_16', 'L1_var_24', 'L1_var_32', 'L1_var_64',
          'L1_quant_16', 'L1_quant_24', 'L1_quant_32', 'L1_quant_64',
          'L2_fix_16', 'L2_fix_24', 'L2_fix_32', 'L2_fix_64',
            'L2_var_16', 'L2_var_24', 'L2_var_32', 'L2_var_64',
            'L2_quant_16', 'L2_quant_24', 'L2_quant_32', 'L2_quant_64']

df = pd.DataFrame([])
for data, name in zip(distances, names):
    df.loc[:,name] = data
    df.loc[:,'lengths'] = lengths_all
    df.to_csv('testResult/Distances_64.csv')