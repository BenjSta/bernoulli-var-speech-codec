from df import init_df
from df.config import config as dfconfig
import h5py
import toml
from third_party.BigVGAN.env import AttrDict
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import scipy.signal
import scipy.interpolate
import tqdm
import os
from ptflops import get_model_complexity_info
import torch
from dataset import load_utterances_easycom, EasycomDataset
from third_party.BigVGAN.meldataset import mel_spectrogram
from metrics import compute_dnsmos, compute_pesq, compute_mean_wacc
import whisper



utts_train, utts_val, _ = load_utterances_easycom(pathconfig["Easycom_root"])


dfmodel, df_state, _ = init_df()  # Load default model
dfmodel = dfmodel.to('cpu')
dfconfig.set("DEVICE", "cpu", str, "train")


valset = EasycomDataset(pathconfig["Easycom_root"], utts_val, fs=48000,
                        duration=None,
                        denoiser=dfmodel, denoiser_state=df_state)


np.random.seed(1)
val_tensorboard_examples = np.random.choice(len(utts_val), 15, replace=False)
np.random.seed()

asr_model = whisper.load_model("medium.en")

config = toml.load(
    "configs_multich_denoiser_training/config_gcrn_densedec.toml")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 'cuda'
vocoder_config_attr_dict = AttrDict(config['vocoder_config'])

# load a vocoder for waveform generation
generator = BigVGAN(vocoder_config_attr_dict).to(device)
print("Generator params: {}".format(sum(p.numel()
      for p in generator.parameters())))

state_dict_g = load_checkpoint(config['vocoder_checkpoint'], device)
convert_vocoder_state_dict(
    state_dict_g, type=config['vocoder_state_dict_type'])

generator.load_state_dict(state_dict_g["generator"])
generator.eval()
print(generator)


def constr(input_res):
    return {
        "x": torch.ones((1, 80, input_res[0])).to(device),
        "length": 1 * config["fs"],
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

log_dir_noisy = os.path.join(config["chkpt_log_dir"], "logs", "noisy")
log_dir_clean = os.path.join(config["chkpt_log_dir"], "logs", "clean")
log_dir_vocoded = os.path.join(config["chkpt_log_dir"], "logs", "vocoded")
log_dir_spear = os.path.join(config["chkpt_log_dir"], "logs", "spear_subfull")


os.makedirs(log_dir_noisy, exist_ok=True)
os.makedirs(log_dir_clean, exist_ok=True)
os.makedirs(log_dir_vocoded, exist_ok=True)
os.makedirs(log_dir_spear, exist_ok=True)

sw_noisy = SummaryWriter(log_dir_noisy)
sw_clean = SummaryWriter(log_dir_clean)
sw_vocoded = SummaryWriter(log_dir_vocoded)
sw_spear = SummaryWriter(log_dir_spear)

mel_spec_config = (config["winsize"],
                   config["num_mels"],
                   config["fs"],
                   config["hopsize"],
                   config["winsize"],
                   config["fmin"],
                   config["fmax"],
                   config["mel_pad_left"])

np.random.seed(1)
y_all = []
y_all_resampled = []
vocoded_all = []
mae_noisy_all = []
noisy_all = []
spear_all = []
transcripts_all = []

for i in tqdm.tqdm(range(len(utts_val))):
    with torch.no_grad():
        (x, _, y, doa, transcript) = valset.__getitem__(i)
        transcripts_all.append(transcript)

        x_16k = scipy.signal.resample_poly(x, 16000, 48000, axis=0)
        y_22k = scipy.signal.resample_poly(y, 22050, 48000, axis=0)

        doa_spear = scipy.interpolate.interp1d(0.05 *
                        np.arange(doa.shape[0]), doa,
                        axis=0, fill_value='extrapolate')(np.arange(
                            np.ceil(x_16k.shape[0] /
                                    spear_config['processing_hopsize'])+1)*
                                    spear_config['processing_hopsize']/16000)
        azi_spear = np.arctan2(doa_spear[:, 0], doa_spear[:, 2])
        ele_spear = np.pi/2 - np.arccos(doa_spear[:, 1] / np.linalg.norm(doa_spear, axis=1))
        doa_spear = np.stack([azi_spear, ele_spear], axis=1) * 180/np.pi
        
        enh_spear = 10 * spearmodel(torch.from_numpy(x_16k[None, :, :].astype('float32')).to('cuda'),
                               torch.from_numpy(doa_spear[None, :, :].astype('float32')).to('cuda')).detach().cpu().numpy()[0, :, 0]

        spear_all.append(enh_spear)
        
        noisy_all.append(10 * x[:, 1])
        

        y_mel = mel_spectrogram(torch.from_numpy(y_22k[None, :, 0]).to(device), *mel_spec_config)

        y_g_hat = generator(y_mel, y.shape[0]).detach().cpu().numpy()[0,0,:]

        vocoded_all.append(y_g_hat)
        y_all.append(y[:, 0])
        y_all_resampled.append(y_22k[:, 0])

        
        


        # y_all.append(y_orig[0, :].detach().cpu().numpy())
        # y_all_resampled.append(y[0, :].detach().cpu().numpy())

lengths_all = np.array([y.shape[0] for y in y_all])

noisy_pesq_all = compute_pesq(y_all, noisy_all, 48000)
clean_pesq_all = compute_pesq(y_all, y_all, 48000)
vocoded_pesq_all = compute_pesq(y_all_resampled, vocoded_all, config["fs"])
spear_pesq_all = compute_pesq(y_all, spear_all, 16000)

noisy_ovr_all, noisy_sig_all, noisy_bak_all = compute_dnsmos(noisy_all, 48000)
clean_ovr_all, clean_sig_all, clean_bak_all = compute_dnsmos(y_all, 48000)
vocoded_ovr_all, vocoded_sig_all, vocoded_bak_all = compute_dnsmos(
    vocoded_all, config["fs"])
spear_ovr_all, spear_sig_all, spear_bak_all = compute_dnsmos(spear_all, 16000)

mean_noisy_pesq = np.mean(
    lengths_all * np.array(noisy_pesq_all)) / np.mean(lengths_all)
mean_clean_pesq = np.mean(
    lengths_all * np.array(clean_pesq_all)) / np.mean(lengths_all)
mean_vocoded_pesq = np.mean(
    lengths_all * np.array(vocoded_pesq_all)) / np.mean(lengths_all)
mean_spear_pesq = np.mean(
    lengths_all * np.array(spear_pesq_all)) / np.mean(lengths_all)

mean_noisy_ovr = np.mean(
    lengths_all * np.array(noisy_ovr_all)) / np.mean(lengths_all)
mean_clean_ovr = np.mean(
    lengths_all * np.array(clean_ovr_all)) / np.mean(lengths_all)
mean_vocoded_ovr = np.mean(
    lengths_all * np.array(vocoded_ovr_all)) / np.mean(lengths_all)
mean_spear_ovr = np.mean(
    lengths_all * np.array(spear_ovr_all)) / np.mean(lengths_all)

mean_noisy_sig = np.mean(
    lengths_all * np.array(noisy_sig_all)) / np.mean(lengths_all)
mean_clean_sig = np.mean(
    lengths_all * np.array(clean_sig_all)) / np.mean(lengths_all)
mean_vocoded_sig = np.mean(
    lengths_all * np.array(vocoded_sig_all)) / np.mean(lengths_all)
mean_spear_sig = np.mean(
    lengths_all * np.array(spear_sig_all)) / np.mean(lengths_all)

mean_noisy_bak = np.mean(
    lengths_all * np.array(noisy_bak_all)) / np.mean(lengths_all)
mean_clean_bak = np.mean(
    lengths_all * np.array(clean_bak_all)) / np.mean(lengths_all)
mean_vocoded_bak = np.mean(
    lengths_all * np.array(vocoded_bak_all)) / np.mean(lengths_all)
mean_spear_bak = np.mean(
    lengths_all * np.array(spear_bak_all)) / np.mean(lengths_all)

wacc_noisy = compute_mean_wacc(
    noisy_all, transcripts_all, 48000, asr_model=asr_model
)
wacc_clean = compute_mean_wacc(
    y_all, transcripts_all, 48000, asr_model=asr_model
)
wacc_vocoded = compute_mean_wacc(
    vocoded_all, transcripts_all, config["fs"], asr_model=asr_model
)
wacc_spear = compute_mean_wacc(
    spear_all, transcripts_all, 16000, asr_model=asr_model
)

sw_noisy.add_scalar("MAE", np.nan, 0)
sw_noisy.add_scalar("PESQ", mean_noisy_pesq, 0)
sw_noisy.add_scalar("DNSMOS-OVR", mean_noisy_ovr, 0)
sw_noisy.add_scalar("DNSMOS-SIG", mean_noisy_sig, 0)
sw_noisy.add_scalar("DNSMOS-BAK", mean_noisy_bak, 0)
sw_noisy.add_scalar("Wacc", wacc_noisy, 0)

sw_clean.add_scalar("MAE", 0.0, 0)
sw_clean.add_scalar("PESQ", mean_clean_pesq, 0)
sw_clean.add_scalar("DNSMOS-OVR", mean_clean_ovr, 0)
sw_clean.add_scalar("DNSMOS-SIG", mean_clean_sig, 0)
sw_clean.add_scalar("DNSMOS-BAK", mean_clean_bak, 0)
sw_clean.add_scalar("Wacc", wacc_clean, 0)

sw_vocoded.add_scalar("MAE", 0.0, 0)
sw_vocoded.add_scalar("PESQ", mean_vocoded_pesq, 0)
sw_vocoded.add_scalar("DNSMOS-OVR", mean_vocoded_ovr, 0)
sw_vocoded.add_scalar("DNSMOS-SIG", mean_vocoded_sig, 0)
sw_vocoded.add_scalar("DNSMOS-BAK", mean_vocoded_bak, 0)
sw_vocoded.add_scalar("Wacc", wacc_vocoded, 0)

sw_spear.add_scalar("MAE", np.nan, 0)
sw_spear.add_scalar("PESQ", mean_spear_pesq, 0)
sw_spear.add_scalar("DNSMOS-OVR", mean_spear_ovr, 0)
sw_spear.add_scalar("DNSMOS-SIG", mean_spear_sig, 0)
sw_spear.add_scalar("DNSMOS-BAK", mean_spear_bak, 0)
sw_spear.add_scalar("Wacc", wacc_spear, 0)

for i in val_tensorboard_examples:
    sw_noisy.add_audio(
        "%d" % i,
        torch.from_numpy(noisy_all[i]),
        global_step=0,
        sample_rate=48000,
    )
    sw_clean.add_audio(
        "%d" % i,
        torch.from_numpy(y_all[i]),
        global_step=0,
        sample_rate=48000,
    )
    sw_vocoded.add_audio(
        "%d" % i,
        torch.from_numpy(vocoded_all[i]),
        global_step=0,
        sample_rate=config["fs"],
    )
    sw_spear.add_audio(
        "%d" % i,
        torch.from_numpy(spear_all[i]),
        global_step=0,
        sample_rate=16000,
    )
