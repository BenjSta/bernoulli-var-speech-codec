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
from metrics import compute_dnsmos, compute_pesq, compute_mean_wacc
import whisper
import soundfile
import time
import tempfile
import subprocess
from transformers import AutoProcessor, EncodecModel

os.environ['CUDA_VISIBLE_DEVICES'] = '1,'

model_id = "facebook/encodec_24khz"
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


def opus(y, fs, bitrate=6):
    with tempfile.NamedTemporaryFile(suffix='.wav', mode='r+') as tmpname1, tempfile.NamedTemporaryFile(suffix='.opus', mode='r+') as tmpname2:
        soundfile.write('%s' % tmpname1.name, y, fs)
        os.system('ffmpeg -y -i %s  -c:a libopus -b:a %dK %s' %
                (tmpname1.name, bitrate, tmpname2.name))
        os.system('ffmpeg -y -i %s -ar %d %s' % (tmpname2.name, fs, tmpname1.name))
        y, _ = soundfile.read('%s' % tmpname1.name)
    return y.astype('float32')


def lyra(y, fs, bitrate=6):
    with tempfile.NamedTemporaryFile(suffix='.wav', mode='r+') as tmpname, tempfile.TemporaryDirectory() as tmpdir:
        soundfile.write(tmpname.name, scipy.signal.resample_poly(y, 16000, fs), 16000)
        cwd = os.getcwd()
        os.chdir(executables['lyra_base'])
        cmd_str = '%s --input_path="%s" --output_dir="%s" --bitrate=%d' % (
            executables['lyra_encoder'], tmpname.name, tmpdir, bitrate)
        print(cmd_str)
        print(subprocess.call(cmd_str, shell=True))
    
        print(subprocess.call('%s --encoded_path="%s/%s.lyra" --output_dir=./ --bitrate=%d' %
            (executables['lyra_decoder'], tmpdir, os.path.split(tmpname.name)[-1][:-len('.wav')], bitrate), shell=True))
        y, _ = soundfile.read(tmpname.name)
        y = scipy.signal.resample_poly(y, fs, 16000)
        os.chdir(cwd)
    return y.astype('float32')


def encodec(y, fs, bandwidth):
    inputs = encodec_processor(raw_audio=scipy.signal.resample_poly(
        y, 24000, fs), return_tensors="pt", sampling_rate=24000)
    outputs = encodec_model(**inputs, bandwidth=bandwidth)
    audio_values = outputs.audio_values
    return scipy.signal.resample_poly(
        audio_values.detach().cpu().numpy()[0, 0, :], fs, 24000)


sw_clean = SummaryWriter(os.path.join(chkpt_log_dirs['chkpt_log_dir'], "clean"))

sw_opus6 = SummaryWriter(os.path.join(chkpt_log_dirs['chkpt_log_dir'], "opus_6k"))
sw_opus10 = SummaryWriter(os.path.join(chkpt_log_dirs['chkpt_log_dir'], "opus_10k"))
sw_opus14 = SummaryWriter(os.path.join(chkpt_log_dirs['chkpt_log_dir'], "opus_14"))

sw_lyra3_2 = SummaryWriter(os.path.join(chkpt_log_dirs['chkpt_log_dir'], "lyra3.2k"))
sw_lyra6 = SummaryWriter(os.path.join(chkpt_log_dirs['chkpt_log_dir'], "lyra6k"))
sw_lyra9_2 = SummaryWriter(os.path.join(chkpt_log_dirs['chkpt_log_dir'], "lyra9.2k"))

sw_encodec1_5 = os.path.join(chkpt_log_dirs['chkpt_log_dir'], "encodec1.5k")
sw_encodec3 = os.path.join(chkpt_log_dirs['chkpt_log_dir'], "encodec3k")
sw_encodec6 = os.path.join(chkpt_log_dirs['chkpt_log_dir'], "encodec6k")
sw_encodec12 = os.path.join(chkpt_log_dirs['chkpt_log_dir'], "encodec12k")

sws = [sw_clean, sw_opus6, sw_opus10, sw_opus14, sw_lyra3_2, sw_lyra6,
       sw_lyra9_2, sw_encodec1_5, sw_encodec3, sw_encodec6, sw_encodec12]


clean_all = []

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

sigs = [clean_all, opus6_all, opus10_all, opus14_all, lyra3_2_all,
        lyra6_all, lyra9_2_all, encodec1_5_all, encodec3_all,
        encodec6_all, encodec12_all]

np.random.seed(1)
for (y,) in tqdm.tqdm(val_dataloader):
    with torch.no_grad():
        clean_all.append(y[0, :].numpy())

        opus6_all.append(opus(y[0, :].numpy(), 48000, 6))
        opus10_all.append(opus(y[0, :].numpy(), 48000, 10))
        opus14_all.append(opus(y[0, :].numpy(), 48000, 14))

        lyra3_2_all.append(lyra(y[0, :].numpy(), 48000, 3200))
        lyra6_all.append(lyra(y[0, :].numpy(), 48000, 6000))
        lyra9_2_all.append(lyra(y[0, :].numpy(), 48000, 9200))

        encodec1_5_all.append(encodec(y[0, :].numpy(), 48000, 1.5))
        encodec3_all.append(encodec(y[0, :].numpy(), 48000, 3))
        encodec6_all.append(encodec(y[0, :].numpy(), 48000, 6))
        encodec12_all.append(encodec(y[0, :].numpy(), 48000, 12))


# lengths_all = np.array([y.shape[0] for y in y_all])

# noisy_pesq_all = compute_pesq(y_all, noisy_all, 48000)
# clean_pesq_all = compute_pesq(y_all, y_all, 48000)
# vocoded_pesq_all = compute_pesq(y_all_resampled, vocoded_all, config["fs"])
# dfnet_pesq_all = compute_pesq(y_all, dfnet_all, 48000)

# noisy_ovr_all, noisy_sig_all, noisy_bak_all = compute_dnsmos(noisy_all, 48000)
# clean_ovr_all, clean_sig_all, clean_bak_all = compute_dnsmos(y_all, 48000)
# vocoded_ovr_all, vocoded_sig_all, vocoded_bak_all = compute_dnsmos(vocoded_all, config["fs"])
# dfnet_ovr_all, dfnet_sig_all, dfnet_bak_all = compute_dnsmos(dfnet_all, 48000)

# mean_noisy_mae = np.mean(lengths_all * np.array(mae_noisy_all)) / np.mean(lengths_all)

# mean_noisy_pesq = np.mean(lengths_all * np.array(noisy_pesq_all)) / np.mean(lengths_all)
# mean_clean_pesq = np.mean(lengths_all * np.array(clean_pesq_all)) / np.mean(lengths_all)
# mean_vocoded_pesq = np.mean(lengths_all * np.array(vocoded_pesq_all)) / np.mean(lengths_all)
# mean_dfnet_pesq = np.mean(lengths_all * np.array(dfnet_pesq_all)) / np.mean(lengths_all)

# mean_noisy_ovr = np.mean(lengths_all * np.array(noisy_ovr_all)) / np.mean(lengths_all)
# mean_clean_ovr = np.mean(lengths_all * np.array(clean_ovr_all)) / np.mean(lengths_all)
# mean_vocoded_ovr = np.mean(lengths_all * np.array(vocoded_ovr_all)) / np.mean(lengths_all)
# mean_dfnet_ovr = np.mean(lengths_all * np.array(dfnet_ovr_all)) / np.mean(lengths_all)

# mean_noisy_sig = np.mean(lengths_all * np.array(noisy_sig_all)) / np.mean(lengths_all)
# mean_clean_sig = np.mean(lengths_all * np.array(clean_sig_all)) / np.mean(lengths_all)
# mean_vocoded_sig = np.mean(lengths_all * np.array(vocoded_sig_all)) / np.mean(lengths_all)
# mean_dfnet_sig = np.mean(lengths_all * np.array(dfnet_sig_all)) / np.mean(lengths_all)

# mean_noisy_bak = np.mean(lengths_all * np.array(noisy_bak_all)) / np.mean(lengths_all)
# mean_clean_bak = np.mean(lengths_all * np.array(clean_bak_all)) / np.mean(lengths_all)
# mean_vocoded_bak = np.mean(lengths_all * np.array(vocoded_bak_all)) / np.mean(lengths_all)
# mean_dfnet_bak = np.mean(lengths_all * np.array(dfnet_bak_all)) / np.mean(lengths_all)

# wacc_noisy = compute_mean_wacc(
#     noisy_all, txt_val, 48000, asr_model=asr_model
# )
# wacc_clean = compute_mean_wacc(
#     y_all, txt_val, 48000, asr_model=asr_model
# )
# wacc_vocoded = compute_mean_wacc(
#     vocoded_all, txt_val, config["fs"], asr_model=asr_model
# )
# wacc_dfnet = compute_mean_wacc(
#     dfnet_all, txt_val, 48000, asr_model=asr_model
# )

# sw_noisy.add_scalar("MAE", mean_noisy_mae, 0)
# sw_noisy.add_scalar("PESQ", mean_noisy_pesq, 0)
# sw_noisy.add_scalar("DNSMOS-OVR", mean_noisy_ovr, 0)
# sw_noisy.add_scalar("DNSMOS-SIG", mean_noisy_sig, 0)
# sw_noisy.add_scalar("DNSMOS-BAK", mean_noisy_bak, 0)
# sw_noisy.add_scalar("Wacc", wacc_noisy, 0)

# sw_clean.add_scalar("MAE", 0.0, 0)
# sw_clean.add_scalar("PESQ", mean_clean_pesq, 0)
# sw_clean.add_scalar("DNSMOS-OVR", mean_clean_ovr, 0)
# sw_clean.add_scalar("DNSMOS-SIG", mean_clean_sig, 0)
# sw_clean.add_scalar("DNSMOS-BAK", mean_clean_bak, 0)
# sw_clean.add_scalar("Wacc", wacc_clean, 0)

# sw_vocoded.add_scalar("MAE", 0.0, 0)
# sw_vocoded.add_scalar("PESQ", mean_vocoded_pesq, 0)
# sw_vocoded.add_scalar("DNSMOS-OVR", mean_vocoded_ovr, 0)
# sw_vocoded.add_scalar("DNSMOS-SIG", mean_vocoded_sig, 0)
# sw_vocoded.add_scalar("DNSMOS-BAK", mean_vocoded_bak, 0)
# sw_vocoded.add_scalar("Wacc", wacc_vocoded, 0)

# sw_dfnet.add_scalar("MAE", np.nan, 0)
# sw_dfnet.add_scalar("PESQ", mean_dfnet_pesq, 0)
# sw_dfnet.add_scalar("DNSMOS-OVR", mean_dfnet_ovr, 0)
# sw_dfnet.add_scalar("DNSMOS-SIG", mean_dfnet_sig, 0)
# sw_dfnet.add_scalar("DNSMOS-BAK", mean_dfnet_bak, 0)
# sw_dfnet.add_scalar("Wacc", wacc_dfnet, 0)

# for sw, sig in zip(sws, sigs):
#     for i in val_tensorboard_examples:
#         sw.add_audio(
#             "%d" % i,
#             torch.from_numpy(sig[i]),
#             global_step=0,
#             sample_rate=48000,
#         )
