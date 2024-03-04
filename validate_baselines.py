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
from metrics import compute_dnsmos, compute_pesq, compute_mean_wacc, compute_mcd
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

os.environ['CUDA_VISIBLE_DEVICES'] = '0,'

model_id = "facebook/encodec_24khz"
vocoder_config = toml.load("configs_vocoder/causal_tiny_bigvgan.toml")
vocoder_chkpt_path = "configs_vocoder/g_2500000"
encodec_model = EncodecModel.from_pretrained(model_id)
encodec_processor = AutoProcessor.from_pretrained(model_id)

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

asr_model = whisper.load_model("medium.en")


validation_dataset = SpeechDataset(
    clean_val,
    duration=None,
    fs=48000,
)

val_dataloader = DataLoader(
    validation_dataset,
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

vocoder_config_attr_dict = AttrDict(vocoder_config['vocoder_config'])


# load a vocoder for waveform generation
generator = BigVGAN(vocoder_config_attr_dict).to('cuda')
print("Generator params: {}".format(sum(p.numel()
      for p in generator.parameters())))

state_dict_g = load_checkpoint(vocoder_chkpt_path, 'cuda')
generator.load_state_dict(state_dict_g["generator"])
generator.eval()
print(generator)


def constr(input_res):
    return {
        "x": torch.ones((1, 80, input_res[0])).to('cuda'),
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


def opus(y, fs, bitrate=6):
    with tempfile.NamedTemporaryFile(suffix='.wav', mode='r+') as tmpfile1, tempfile.NamedTemporaryFile(suffix='.opus', mode='r+') as tmpfile2:
        soundfile.write('%s' % tmpfile1.name, y, fs)
        subprocess.call('ffmpeg -y -i %s  -c:a libopus -b:a %dK %s' %
                        (tmpfile1.name, bitrate,
                         tmpfile2.name), stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT, shell=True)
        subprocess.call('ffmpeg -y -i %s -ar %d %s' %
                        (tmpfile2.name, fs, tmpfile1.name), stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT, shell=True)
        y, _ = soundfile.read('%s' % tmpfile1.name)
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


sw_clean = SummaryWriter(os.path.join(
    chkpt_log_dirs['chkpt_log_dir'], "clean"))

sw_vocoded = SummaryWriter(os.path.join(
    chkpt_log_dirs['chkpt_log_dir'], "vocoder_causal"))

sw_opus6 = SummaryWriter(os.path.join(
    chkpt_log_dirs['chkpt_log_dir'], "opus_6k"))
sw_opus10 = SummaryWriter(os.path.join(
    chkpt_log_dirs['chkpt_log_dir'], "opus_10k"))
sw_opus14 = SummaryWriter(os.path.join(
    chkpt_log_dirs['chkpt_log_dir'], "opus_14"))

sw_lyra3_2 = SummaryWriter(os.path.join(
    chkpt_log_dirs['chkpt_log_dir'], "lyra3.2k"))
sw_lyra6 = SummaryWriter(os.path.join(
    chkpt_log_dirs['chkpt_log_dir'], "lyra6k"))
sw_lyra9_2 = SummaryWriter(os.path.join(
    chkpt_log_dirs['chkpt_log_dir'], "lyra9.2k"))

sw_encodec1_5 = SummaryWriter(os.path.join(
    chkpt_log_dirs['chkpt_log_dir'], "encodec1.5k"))
sw_encodec3 = SummaryWriter(os.path.join(
    chkpt_log_dirs['chkpt_log_dir'], "encodec3k"))
sw_encodec6 = SummaryWriter(os.path.join(
    chkpt_log_dirs['chkpt_log_dir'], "encodec6k"))
sw_encodec12 = SummaryWriter(os.path.join(
    chkpt_log_dirs['chkpt_log_dir'], "encodec12k"))


clean_all = []

vocoded_all = []

opus6_all = []
opus10_all = []
opus14_all = []

lyra3_2_all = []
lyra6_all = []
lyra9_2_all = []

encodec1_5_all = []
encodec3_all = []
encodec6_all = []
encodec12_all = []

sws = [sw_clean, sw_vocoded, sw_opus6, sw_opus10, sw_opus14, sw_lyra3_2, sw_lyra6,
       sw_lyra9_2, sw_encodec1_5, sw_encodec3, sw_encodec6, sw_encodec12]

sigs = [clean_all, vocoded_all, opus6_all, opus10_all, opus14_all, lyra3_2_all,
        lyra6_all, lyra9_2_all, encodec1_5_all, encodec3_all,
        encodec6_all, encodec12_all]

np.random.seed(1)
for (y,) in tqdm.tqdm(val_dataloader):
    with torch.no_grad():
        clean_all.append(y[0, :].numpy())

        y_resampled = scipy.signal.resample_poly(
            y.cpu().numpy()[0, :], vocoder_config['fs'], 48000)
        y_mel = mel_spectrogram(torch.from_numpy(y_resampled).to('cuda')[
                                None, :], **mel_spec_config)
        y_vocoded = generator(y_mel, y_resampled.shape[0])[
            0, :].detach().cpu().numpy()
        y_vocoded = scipy.signal.resample_poly(
            y_vocoded.cpu().numpy()[0, :], 48000, vocoder_config['fs'])

        vocoded_all.append(y_vocoded)

        opus6_all.append(10**(-10/20)*opus(10**(10/20)
                         * y[0, :].numpy(), 48000, 6))
        opus10_all.append(10**(-10/20)*opus(10**(10/20)
                          * y[0, :].numpy(), 48000, 10))
        opus14_all.append(10**(-10/20)*opus(10**(10/20)
                          * y[0, :].numpy(), 48000, 14))

        lyra3_2_all.append(10**(-10/20)*lyra(10**(10/20)
                           * y[0, :].numpy(), 48000, 3200))
        lyra6_all.append(10**(-10/20)*lyra(10**(10/20)
                         * y[0, :].numpy(), 48000, 6000))
        lyra9_2_all.append(10**(-10/20)*lyra(10**(10/20)
                           * y[0, :].numpy(), 48000, 9200))

        encodec1_5_all.append(
            10**(-10/20)*encodec(10**(10/20)*y[0, :].numpy(), 48000, 1.5))
        encodec3_all.append(
            10**(-10/20)*encodec(10**(10/20)*y[0, :].numpy(), 48000, 3))
        encodec6_all.append(
            10**(-10/20)*encodec(10**(10/20)*y[0, :].numpy(), 48000, 6))
        encodec12_all.append(
            10**(-10/20)*encodec(10**(10/20)*y[0, :].numpy(), 48000, 12))


lengths_all = np.array([y.shape[0] for y in clean_all])

for sw, sigs_method in zip(sws, sigs):
    pesq = compute_pesq(clean_all, sigs_method, 48000)
    ovr, sig, bak = compute_dnsmos(sigs_method, 48000)
    mcd = compute_mcd(clean_all, sigs_method, 48000)

    mean_pesq = np.mean(lengths_all * np.array(pesq)) / np.mean(lengths_all)
    mean_sig = np.mean(lengths_all * np.array(sig)) / np.mean(lengths_all)
    mean_mcd = np.mean(lengths_all * np.array(mcd)) / np.mean(lengths_all)

    mean_wacc = compute_mean_wacc(sigs_method, txt_val, 48000, asr_model)

    sw.add_scalar('PESQ', mean_pesq, 0)
    sw.add_scalar('SIG', mean_sig, 0)
    sw.add_scalar('MCD', mean_mcd, 0)
    sw.add_scalar('WAcc', mean_wacc, 0)

    for i in val_tensorboard_examples:
        sw.add_audio(
            "%d" % i,
            torch.from_numpy(sigs_method[i]),
            global_step=0,
            sample_rate=48000,
        )
