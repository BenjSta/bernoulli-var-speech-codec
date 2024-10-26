import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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
import torch
# from metrics import compute_dnsmos, compute_pesq, compute_mean_wacc, compute_nisqa, compute_mcd, compute_estimated_metrics, compute_visqol
import whisper
import soundfile
import time
import tempfile
import subprocess
import glob
from transformers import AutoProcessor, EncodecModel
from third_party.BigVGAN.models import BigVGAN
from third_party.BigVGAN.utils import load_checkpoint
from third_party.BigVGAN.env import AttrDict
from third_party.BigVGAN.meldataset import mel_spectrogram
from ptflops import get_model_complexity_info
from bvrnn import BVRNN
import pandas as pd
import dac
from audiotools import AudioSignal
from third_party.AudioDec.utils.audiodec import AudioDec, assign_model



MAKEUP_GAIN = 10.0

wavPath = './wavdemo/'


model_id = "facebook/encodec_24khz"
encodec_model = EncodecModel.from_pretrained(model_id)
encodec_processor = AutoProcessor.from_pretrained(model_id)
# config = toml.load("configs_coding/config_32bit.toml")
config = toml.load("configs_coding/config_varBitRate.toml")


# Download a model
model_path = dac.utils.download(model_type="44khz", model_bitrate="8kbps")
dac_model44 = dac.DAC.load(model_path)#+
dac_model44.eval()

model_path = dac.utils.download(model_type="24khz", model_bitrate="8kbps")
dac_model24 = dac.DAC.load(model_path)
dac_model24.eval()

model_path = dac.utils.download(model_type="16khz", model_bitrate="8kbps")
dac_model16 = dac.DAC.load(model_path)
dac_model16.eval()



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


sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model("libritts_v1")
encoder_checkpoint = os.path.join("third_party/AudioDec", encoder_checkpoint)
decoder_checkpoint = os.path.join("third_party/AudioDec", decoder_checkpoint)
audiodec_model = AudioDec(tx_device=device, rx_device=device)
audiodec_model.load_transmitter(encoder_checkpoint)
audiodec_model.load_receiver(encoder_checkpoint, decoder_checkpoint)

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

test_229 = [c for c in clean_test if 'p229' in c and soundfile.info(c).duration > 3.7 and soundfile.info(c).duration < 4.3]
test_230 = [c for c in clean_test if 'p230' in c and soundfile.info(c).duration > 3.7 and soundfile.info(c).duration < 4.3]
test_232 = [c for c in clean_test if 'p232' in c and soundfile.info(c).duration > 3.7 and soundfile.info(c).duration < 4.3]
test_237 = [c for c in clean_test if 'p237' in c and soundfile.info(c).duration > 3.7 and soundfile.info(c).duration < 4.3]


np.random.seed(11)
test_229 = np.random.choice(test_229)
test_230 = np.random.choice(test_230)
test_232 = np.random.choice(test_232)
test_237 = np.random.choice(test_237)

avspeech_all = glob.glob('/media/DATAslow/shared/stahl/mtedx_iwslt2021/48k/**/*.wav')
avspeech_all = np.random.choice(avspeech_all, 4)
avspeech1 = avspeech_all[0]
avspeech2 = avspeech_all[1]
avspeech3 = avspeech_all[2]
avspeech4 = avspeech_all[3]



os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_id = "facebook/encodec_24khz"
vocoder_config = toml.load(
    "configs_vocoder_training/config_vocoder_causal_snake_doubleseg.toml")
vocoder_config_attr_dict = AttrDict(vocoder_config)

vocoder_chkpt_path = "pretrained_and_finetuned_vocoder_chkpts/bigvgan_causal_tiny_g_2500000"
vocoder_chkpt_path_ft = "pretrained_and_finetuned_vocoder_chkpts/bigvgan_causal_tiny_ftbvrnn_g_3500000"
vocoder_chkpt_path_ftvq = "pretrained_and_finetuned_vocoder_chkpts/bigvgan_causal_tiny_ftvq_g_3500000"


vocoder_config_bigger = toml.load(
    "configs_vocoder_training/config_vocoder_causal_snake_doubleseg_bigger.toml")
vocoder_config_bigger_attr_dict = AttrDict(vocoder_config_bigger)
vocoder_chkpt_path_bigger = "pretrained_and_finetuned_vocoder_chkpts/bigvgan_base_causal"
vocoder_config_bigger_sym = toml.load(
    "configs_vocoder_training/config_vocoder_non_causal_snake_bigger.toml")
vocoder_chkpt_path_bigger_sym = "pretrained_and_finetuned_vocoder_chkpts/bigvgan_base_sym"
vocoder_config_bigger_sym_attr_dict = AttrDict(vocoder_config_bigger_sym)

encodec_model = EncodecModel.from_pretrained(model_id)
encodec_processor = AutoProcessor.from_pretrained(model_id)

encodec_model.forward = encodec_model.encode
def constr(input_res):
    return {
        "input_values": torch.ones((1, 1, input_res[0])).to('cpu'),
        "padding_mask": torch.ones((1, input_res[0])).to('cpu'),
    }
macs, params = get_model_complexity_info(
    encodec_model,
    (10 * 24000,),
    input_constructor=constr,
    as_strings=False,
    print_per_layer_stat=True,
    verbose=True,
)
print("Computational complexity of Encodec encoder: %g" % macs)
print("Number of parameters in Encodec: %g" % params)

encoder_outputs  = encodec_model.encode(input_values= torch.ones((1, 1, 10*24000)).to('cpu'),
        padding_mask = torch.ones((1, 10*24000)).to('cpu'))

encodec_model.forward = encodec_model.decode
def constr(input_res):
    return {
        "audio_codes": encoder_outputs["audio_codes"],
        "audio_scales": encoder_outputs["audio_scales"],
        "padding_mask": torch.ones((1, input_res[0])).to('cpu'),
    }
macs, params = get_model_complexity_info(
    encodec_model,
    (10 * 24000,),
    input_constructor=constr,
    as_strings=False,
    print_per_layer_stat=True,
    verbose=True,
)
print("Computational complexity of Encodec decoder: %g" % macs)
print("Number of parameters in Encodec: %g" % params)

print(sum([p.numel() for _, p in encodec_model.quantizer.state_dict().items()]))

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
    (int(10*22050/256),),
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
    outputs = encodec_model(inputs["input_values"], inputs["padding_mask"], bandwidth=bandwidth)
    audio_values = outputs.audio_values
    return scipy.signal.resample_poly(
        audio_values.detach().cpu().numpy()[0, 0, :], fs, 24000)


def dac_fn(y, fs, model, fs_model, bitrate_factor=1.0):
    with tempfile.NamedTemporaryFile(suffix='.wav', mode='r+') as tmpfile:
        if fs != fs_model:
            y = scipy.signal.resample_poly(y, fs_model, fs)
        soundfile.write(tmpfile.name, y, fs_model)
        signal = AudioSignal(tmpfile.name, fs_model)
        signal.to(model.device)
        x = model.preprocess(signal.audio_data, signal.sample_rate)
        #x = dac.preprocess(signal.audio_data, signal.sample_rate)
        n_quantizers = int(np.round(bitrate_factor * model.n_codebooks))
        print("Using DAC at bitrate %g kbps" % (n_quantizers / model.n_codebooks * 8))
        z, _, _, _, _ = model.encode(x, n_quantizers)
        y = model.decode(z)
        y = scipy.signal.resample_poly(y[0,0,:].detach().cpu().numpy(), fs, fs_model)

    return y.astype('float32')

def audiodec(y, fs):
    with tempfile.NamedTemporaryFile(suffix='.wav', mode='r+') as tmpfile:
        if fs != 24000:
            y = scipy.signal.resample_poly(y, 24000, fs)
        
        y = torch.from_numpy(y.astype('float32')).to(device)
        l = y.shape[0]
        y = y[None, None, :]
        y = audiodec_model.tx_encoder.encode(y)
        y = audiodec_model.tx_encoder.quantize(y)
        y = audiodec_model.rx_encoder.lookup(y)
        y = audiodec_model.decoder.decode(y)[0, 0, :l].detach().cpu().numpy()
        
        if fs != 24000:
            y = scipy.signal.resample_poly(y, fs, 24000)

    return y.astype('float32')

for filepath, speaker in zip ([avspeech1, avspeech2, avspeech3, avspeech4],
                              ['av1', 'av2', 'av3', 'av4']):
    with torch.no_grad():
            y48, fs_speech = soundfile.read(filepath)
            print(fs_speech)
            if fs_speech != 48000:
                if y48.ndim == 2:
                    y48 = y48[:, 0] 
                y48 = scipy.signal.resample_poly(y48, 48000, fs_speech)
                fs_speech = 48000
            
            y48 =  10**(-MAKEUP_GAIN/20) * y48.astype('float32')
            assert fs_speech == 48000, "unexpected sampling rate"
            y22 = scipy.signal.resample_poly(y48, vocoder_config['fs'], 48000)
            y24 = scipy.signal.resample_poly(y48, 24000, 48000)
            y16 = scipy.signal.resample_poly(y48, 16000, 48000)

            
            soundfile.write(wavPath + speaker + '.flac', 10**(MAKEUP_GAIN/20) * y48, 48000)

        
            y_mel = mel_spectrogram(torch.from_numpy(y22).to(device)[
                                    None, :], **mel_spec_config)
            y_mel_sym = mel_spectrogram(torch.from_numpy(y22).to(device)[
                                        None, :], **mel_spec_config_sym)

            y_g_hat = generator(y_mel, y22.shape[0])
            soundfile.write(wavPath + speaker + '_bigvgan_tiny_causal.flac',
                            10**(MAKEUP_GAIN/20)*y_g_hat[0, 0, :].detach().cpu().numpy(), config['fs'])
            y_g_hat_big = generator_big(y_mel, y22.shape[0])
            soundfile.write(wavPath + speaker + '_bigvgan_base_causal.flac', 
                            10**(MAKEUP_GAIN/20)*y_g_hat_big[0, 0, :].detach().cpu().numpy(), config['fs'])
            y_g_hat_bigsym = generator_bigsym(y_mel_sym, y22.shape[0])
            soundfile.write(wavPath + speaker + '_bigvgan_base.flac', 
                            10**(MAKEUP_GAIN/20)*y_g_hat_bigsym[0, 0, :].detach().cpu().numpy(), config['fs'])
            

            # variable bitrate
            for varBit in [16, 32, 64]:
                varBit_T = varBit * \
                    torch.ones(y_mel.shape[0], y_mel.shape[2]).to(device)
                
                y_mel_reconst, kld = bvrnn_var(
                    y_mel.permute(0, 2, 1), 1.0, True, varBit_T)
                y_mel_reconst = y_mel_reconst.permute(0, 2, 1)

                y_mel_reconst_sym, kld = bvrnn_var(
                    y_mel_sym.permute(0, 2, 1), 1.0, True, varBit_T)
                y_mel_reconst_sym = y_mel_reconst_sym.permute(0, 2, 1)

                y_g_hat = generator(y_mel_reconst, y22.shape[0])
                y_g_hat_ft = generator_ft(y_mel_reconst, y22.shape[0])
                y_g_hat_big = generator_big(y_mel_reconst, y22.shape[0])
                y_g_hat_bigsym = generator_bigsym(y_mel_reconst_sym, y22.shape[0])

                soundfile.write(wavPath + speaker + '_var%dbit_bigvgan_tiny_causal.flac' % varBit,
                                10**(MAKEUP_GAIN/20)*y_g_hat[0, 0, :].detach().cpu().numpy(), config['fs'])
                soundfile.write(wavPath + speaker + '_var%dbit_finetuned.flac' % varBit,
                                10**(MAKEUP_GAIN/20)*y_g_hat_ft[0, 0, :].detach().cpu().numpy(), config['fs'])
                soundfile.write(wavPath + speaker + '_var%dbit_bigvgan_base_causal.flac' % varBit,
                                10**(MAKEUP_GAIN/20)*y_g_hat_big[0, 0, :].detach().cpu().numpy(), config['fs'])
                soundfile.write(wavPath + speaker + '_var%dbit_bigvgan_base.flac' % varBit,
                                10**(MAKEUP_GAIN/20)*y_g_hat_bigsym[0, 0, :].detach().cpu().numpy(), config['fs'])
                
                soundfile.write(wavPath + speaker + '_var%dbit_bigvgan_tiny_causal_16kHz.flac' % varBit,
                                10**(MAKEUP_GAIN/20)*scipy.signal.resample_poly(
                                    y_g_hat[0, 0, :].detach().cpu().numpy(), 16000, config['fs']), 16000)
                soundfile.write(wavPath + speaker + '_var%dbit_finetuned_16kHz.flac' % varBit,
                                10**(MAKEUP_GAIN/20)*scipy.signal.resample_poly(
                                    y_g_hat_ft[0, 0, :].detach().cpu().numpy(), 16000, config['fs']), 16000)
                soundfile.write(wavPath + speaker + '_var%dbit_bigvgan_base_causal_16kHz.flac' % varBit,
                                10**(MAKEUP_GAIN/20)*scipy.signal.resample_poly(
                                    y_g_hat_big[0, 0, :].detach().cpu().numpy(), 16000, config['fs']), 16000)
                soundfile.write(wavPath + speaker + '_var%dbit_bigvgan_base_16kHz.flac' % varBit,
                                10**(MAKEUP_GAIN/20)*scipy.signal.resample_poly(
                                    y_g_hat_bigsym[0, 0, :].detach().cpu().numpy(), 16000, config['fs']), 16000)
                

            for layers, bitrate in zip([6, 12, 24], [1.38, 2.76, 5.51]):
                
                y_mel_reconst = quantizer_encode_decode(y_mel[0, :, :].cpu().numpy().T, layers)
                y_mel_reconst = torch.from_numpy(y_mel_reconst[None, :, :].astype('float32')).permute(0, 2, 1).to(device)
                
                y_g_hat = generator_ftvq(y_mel_reconst, y22.shape[0])

                soundfile.write(wavPath + speaker + '_vq%g_finetuned.flac' % bitrate,
                                10**(MAKEUP_GAIN/20)*y_g_hat[0, 0, :].detach().cpu().numpy(), config['fs'])
    
  
            #fixed bitrate
            for zdim, model in zip([16,32,64], [bvrnn_fix_16, bvrnn_fix_32, bvrnn_fix_64]):
                varBit = zdim
                varBit_T = varBit * torch.ones(y_mel.shape[0],y_mel.shape[2]).to(device)
                y_mel_reconst, kld = model(y_mel.permute(0, 2, 1), 1.0, True, varBit_T)
                y_mel_reconst = y_mel_reconst.permute(0, 2, 1)
                y_g_hat = generator(y_mel_reconst, y22.shape[0])
                soundfile.write(wavPath + speaker + '_fixed%dbit.flac' % zdim,
                                10**(MAKEUP_GAIN/20)*y_g_hat[0, 0, :].detach().cpu().numpy(), config['fs'])

            
            opus6 = opus(10**(MAKEUP_GAIN/20)* y48, 48000, 6)
            soundfile.write(wavPath + speaker + '_opus6.flac', opus6, 48000)
            opus10 = opus(10**(MAKEUP_GAIN/20)* y48, 48000, 10)
            soundfile.write(wavPath + speaker + '_opus10.flac', opus10, 48000)

            lyra32 = lyra(10**(MAKEUP_GAIN/20)* y16, 16000, 3200)
            soundfile.write(wavPath + speaker + '_lyra3.2.flac', lyra32, 16000)
            lyra6 = lyra(10**(MAKEUP_GAIN/20)* y16, 16000, 6000)
            soundfile.write(wavPath + speaker + '_lyra6.flac', lyra6, 16000)

            encodec_15 = encodec(10**(MAKEUP_GAIN/20)* y24, 24000, 1.5)
            soundfile.write(wavPath + speaker + '_encodec1.5.flac', encodec_15, 24000)
            encodec_6 = encodec(10**(MAKEUP_GAIN/20)* y24, 24000, 6)
            soundfile.write(wavPath + speaker + '_encodec6.flac', encodec_6, 24000)
            
            # dac_16 = dac_fn(10**(MAKEUP_GAIN/20)* y16, 16000, dac_model16, 16000)
            # soundfile.write(wavPath + speaker + '_dac16.flac', dac_16, 16000)
            # dac_24 = dac_fn(10**(MAKEUP_GAIN/20)* y24, 24000, dac_model24, 24000, 0.35)
            # soundfile.write(wavPath + speaker + '_dac24.flac', dac_24, 24000)
            # dac_44 = dac_fn(10**(MAKEUP_GAIN/20)* y48, 48000, dac_model44, 44100)
            # soundfile.write(wavPath + speaker + '_dac44.flac', dac_44, 48000)

            audiodec_24 = audiodec(10**(MAKEUP_GAIN/20)* y24, 24000)
            soundfile.write(wavPath + speaker + '_audiodec24.flac', audiodec_24, 24000)
            
            # opus6_all.append(10**(-MAKEUP_GAIN/20)*opus(10**(MAKEUP_GAIN/20)* y[0, :].numpy(), 48000, 6))
            # opus8_all.append(10**(-MAKEUP_GAIN/20)*opus(10**(MAKEUP_GAIN/20)* y[0, :].numpy(), 48000, 8))
            # opus10_all.append(10**(-MAKEUP_GAIN/20)*opus(10**(MAKEUP_GAIN/20)* y[0, :].numpy(), 48000, 10))
            # opus12_all.append(10**(-MAKEUP_GAIN/20)*opus(10**(MAKEUP_GAIN/20)* y[0, :].numpy(), 48000, 12))

            # lyra3_2_all.append(10**(-MAKEUP_GAIN/20)*lyra(10**(MAKEUP_GAIN/20)* y[0, :].numpy(), 48000, 3200))
            # lyra6_all.append(10**(-MAKEUP_GAIN/20)*lyra(10**(MAKEUP_GAIN/20) * y[0, :].numpy(), 48000, 6000))
            # lyra9_2_all.append(10**(-MAKEUP_GAIN/20)*lyra(10**(MAKEUP_GAIN/20)* y[0, :].numpy(), 48000, 9200))

            # encodec1_5_all.append(10**(-MAKEUP_GAIN/20)*encodec(10**(MAKEUP_GAIN/20)*y[0, :].numpy(), 48000, 1.5))
            # encodec3_all.append(10**(-MAKEUP_GAIN/20)*encodec(10**(MAKEUP_GAIN/20)*y[0, :].numpy(), 48000, 3))
            # encodec6_all.append(10**(-MAKEUP_GAIN/20)*encodec(10**(MAKEUP_GAIN/20)*y[0, :].numpy(), 48000, 6))
            # encodec12_all.append(10**(-MAKEUP_GAIN/20)*encodec(10**(MAKEUP_GAIN/20)*y[0, :].numpy(), 48000, 12))