import toml
import whisper
from metrics import compute_dnsmos, compute_pesq, compute_mean_wacc, compute_mcd
from third_party.BigVGAN.env import AttrDict
import platform
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import scipy.signal
import tqdm
import os
import itertools
from ptflops import get_model_complexity_info
import torch
from third_party.BigVGAN.meldataset import mel_spectrogram
from collections import OrderedDict

from third_party.BigVGAN.models import (
    BigVGAN,
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
)

from third_party.BigVGAN.utils import load_checkpoint, save_checkpoint
from dataset import NoisySpeechDataset, load_paths
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

torch.backends.cudnn.benchmark = True

# LOAD CONFIG
config = toml.load("configs/config_vocoder_bigvgan_base.toml")
config_attr_dict = AttrDict(config)
os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

pathconfig = toml.load("directories.toml")
paths = load_paths(pathconfig["DNS4_root"], pathconfig["VCTK_txt_root"])
clean_train, clean_val, _ = paths["clean"]
_, txt_val, _ = paths["txt"]
noise_train, noise_val, _ = paths["noise"]
rir_train, rir_val, _ = paths["rir"]


np.random.seed(1)
val_tensorboard_examples = np.random.choice(len(clean_val), 15, replace=False)
np.random.seed()


asr_model = whisper.load_model("medium.en")


trainset = NoisySpeechDataset(
    clean_train,
    noise_train,
    rir_train,
    duration=config["train_seq_duration"],
    snr_mu_and_sigma=(5, 10),
    apply_noise=False,
    rir_percentage=0.75,
    fs=22050,
)

if platform.system() == "Linux":
    def numpy_random_seed(ind=None):
        np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))

    worker_init_fn = numpy_random_seed
else:
    worker_init_fn = None


train_dataloader = DataLoader(
    trainset,
    config["batch_size"],
    shuffle=True,
    num_workers=config["num_workers"],
    drop_last=True,
    worker_init_fn=worker_init_fn,
    prefetch_factor=8
)


# define BigVGAN generator
generator = BigVGAN(config_attr_dict).to(device)
print("Generator params: {}".format(sum(p.numel()
      for p in generator.parameters())))

# define discriminators. MPD is used by default
mpd = MultiPeriodDiscriminator(config_attr_dict).to(device)
print("Discriminator mpd params: {}".format(
    sum(p.numel() for p in mpd.parameters())))

# define additional discriminators. BigVGAN uses MRD as default
mrd = MultiResolutionDiscriminator(config_attr_dict).to(device)

print("Discriminator mrd params: {}".format(
    sum(p.numel() for p in mrd.parameters())))

# create or scan the latest checkpoint from checkpoints directory
print(generator)


chkpt_dir = os.path.join(
    config["chkpt_log_dir"], "checkpoints", config["train_name"])
os.makedirs(chkpt_dir, exist_ok=True)


optim_g = torch.optim.AdamW(
    generator.parameters(),
    config_attr_dict.learning_rate,
    betas=[config_attr_dict.adam_b1, config_attr_dict.adam_b2],
)
optim_d = torch.optim.AdamW(
    itertools.chain(mrd.parameters(), mpd.parameters()),
    config_attr_dict.learning_rate,
    betas=[config_attr_dict.adam_b1, config_attr_dict.adam_b2],
)


steps = 0
if config["resume"]:
    state_dict_g = load_checkpoint(os.path.join(chkpt_dir, "g_latest"), device)
    state_dict_do = load_checkpoint(
        os.path.join(chkpt_dir, "do_latest"), device)
    generator.load_state_dict(state_dict_g["generator"])
    optim_g.load_state_dict(state_dict_do["optim_g"])
    optim_d.load_state_dict(state_dict_do["optim_d"])
    mpd.load_state_dict(state_dict_do["mpd"])
    mrd.load_state_dict(state_dict_do["mrd"])
    steps = state_dict_do["steps"]

if "hifigan_chkpt" in config:
    assert config['validate_only'], 'HifiGAN checkpoint only for validation'

    state_dict_g = load_checkpoint(config["hifigan_chkpt"], device)

    state_dict_g["generator"] = OrderedDict(
        (k[len("module."):] if ("module." in k) else k, v)
        for k, v in state_dict_g["generator"].items()
    )

    state_dict_g["generator"] = OrderedDict(
        ((k[:5] + ".1" + k[7:]) if ("ups." in k) else k, v)
        for k, v in state_dict_g["generator"].items()
    )
    generator.load_state_dict(state_dict_g["generator"])


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
print("Computational complexity of vocoder model: %g" % (macs / 1))
print("Number of parameters in vocoder model: %g" % params)

validation_dataset = NoisySpeechDataset(
    clean_val,
    noise_val,
    rir_val,
    duration=None,
    snr_mu_and_sigma=(0, 0),
    apply_noise=True,
    rir_percentage=0.75,
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

log_dir = os.path.join(config["chkpt_log_dir"], "logs", config["train_name"])
os.makedirs(log_dir, exist_ok=True)

sw_proposed = SummaryWriter(log_dir)

mel_spec_config = (config["winsize"],
                   config["num_mels"],
                   config["fs"],
                   config["hopsize"],
                   config["winsize"],
                   config["fmin"],
                   config["fmax"],
                   config["mel_pad_left"])

mel_spec_config_loss = (config["winsize"],
                        config["msl_num_mels"],
                        config["fs"],
                        config["hopsize"],
                        config["winsize"],
                        config["msl_fmin"],
                        config["msl_fmax"],
                        config["mel_pad_left"])


def validate(step):
    generator.eval()
    mpd.eval()
    mrd.eval()

    np.random.seed(1)

    y_all = []
    reconst_all = []

    for batch in tqdm.tqdm(val_dataloader):
        (y, m) = batch
        y = torch.from_numpy(
            scipy.signal.resample_poly(
                y.detach().cpu().numpy(), config["fs"], 48000, axis=1
            )
        )
        y = y.to(device, non_blocking=True)
        valid_length = y.shape[1] // config["hopsize"] * config["hopsize"]
        y = y[:, :valid_length]

        if "hifigan_chkpt" in config:
            y *= 10**(5/10)

        y_mel = mel_spectrogram(y, *mel_spec_config)

        if config["val_clean_ref"]:
            y_g_hat = y[:, None, :]
        else:
            with torch.no_grad():
                y_g_hat = generator(y_mel, y.shape[1])

        if "hifigan_chkpt" in config:
            y_g_hat *= 10**(-5/10)
            y *= 10**(-5/10)

        reconst_all.append(y_g_hat[0, 0, :].detach().cpu().numpy())
        y_all.append(y[0, :].detach().cpu().numpy())

    lengths_all = np.array([y.shape[0] for y in y_all])
    mcd_all = compute_mcd(y_all, reconst_all, config['fs'])
    pesq_all = compute_pesq(y_all, reconst_all, config["fs"])
    _, sig_all, _ = compute_dnsmos(reconst_all, config["fs"])

    np.savetxt(
        os.path.join(log_dir, "metrics%d.csv" % step),
        np.stack((pesq_all, sig_all, mcd_all)).T,
        fmt="%.3f",
        delimiter=",",
        header="PESQ-WB, DNSMOS-SIG, MCD",

    )

    mcd_mean = np.mean(lengths_all * mcd_all) / np.mean(lengths_all)
    pesq_mean = np.mean(lengths_all * pesq_all) / np.mean(lengths_all)
    sig_mean = np.mean(lengths_all * sig_all) / np.mean(lengths_all)

    wacc_proposed = compute_mean_wacc(
        reconst_all, txt_val, config["fs"], asr_model=asr_model
    )

    sw_proposed.add_scalar("MCD", mcd_mean, step)
    sw_proposed.add_scalar("PESQ-WB", pesq_mean, step)
    sw_proposed.add_scalar("DNSMOS-SIG", sig_mean, step)
    sw_proposed.add_scalar("Wacc", wacc_proposed, step)

    for i in val_tensorboard_examples:
        sw_proposed.add_audio(
            "%d" % i,
            torch.from_numpy(reconst_all[i]),
            global_step=step,
            sample_rate=config["fs"],
        )
    np.random.seed()

    generator.train()
    mpd.train()
    mrd.train()


def save_chkpts(g_path, do_path):
    save_checkpoint(
        g_path,
        {
            "generator": generator.state_dict(),
            "steps": steps,
        },
    )
    save_checkpoint(
        do_path,
        {
            "mpd": mpd.state_dict(),
            "mrd": mrd.state_dict(),
            "optim_g": optim_g.state_dict(),
            "optim_d": optim_d.state_dict(),
            "steps": steps,
        },
    )


if config["validate_only"]:
    validate(steps)
    exit()


# main training loop
generator.train()
mpd.train()
mrd.train()

while True:
    continue_next_epoch = True
    pbar = tqdm.tqdm(train_dataloader)
    for batch in pbar:
        (y, m) = batch
        y = y.to(device, non_blocking=True)
        valid_length = y.shape[1] // config["hopsize"] * config["hopsize"]
        y = y[:, :valid_length]

        y_mel = mel_spectrogram(y, *mel_spec_config)
        y_mel_for_loss = mel_spectrogram(10**(10/20)*y, *mel_spec_config_loss)

        # multiply y and y_g_hat to scale up
        y_g_hat = 10**(10/20)*generator(y_mel, y.shape[1])
        y = 10**(10/20)*y

        y_g_hat_mel = mel_spectrogram(
            y_g_hat.squeeze(1), *mel_spec_config_loss)

        optim_d.zero_grad()

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = mpd(y[:, None, :], y_g_hat.detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(
            y_df_hat_r, y_df_hat_g
        )

        # MRD
        y_ds_hat_r, y_ds_hat_g, _, _ = mrd(y[:, None, :], y_g_hat.detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(
            y_ds_hat_r, y_ds_hat_g
        )

        loss_disc_all = loss_disc_s + loss_disc_f

        loss_disc_all.backward()
        grad_norm_mpd = torch.nn.utils.clip_grad_norm_(
            mpd.parameters(), 1000.0)
        grad_norm_mrd = torch.nn.utils.clip_grad_norm_(
            mrd.parameters(), 1000.0)
        optim_d.step()

        # generator
        optim_g.zero_grad()

        # L1 Mel-Spectrogram Loss
        loss_mel = F.l1_loss(y_mel_for_loss, y_g_hat_mel) * 45

        # MPD loss
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(
            y[:, None, :], y_g_hat)
        loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
        loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)

        # MRD loss
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = mrd(
            y[:, None, :], y_g_hat)
        loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

        loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

        loss_gen_all.backward()
        grad_norm_g = torch.nn.utils.clip_grad_norm_(
            generator.parameters(), 1000.0)
        optim_g.step()

        steps += 1
        # # STDOUT logging
        if steps % 10 == 0:
            printstr = (
                "Steps : {:d}, GGradNorm: {:4.3f}, Gen Loss Total : {:4.3f}, Disc Loss Total : {:4.3f}, Mel-Spec. Loss : {:4.3f}".format(
                    steps,
                    grad_norm_g,
                    loss_gen_all,
                    loss_disc_all,
                    loss_mel,
                )
            )
            pbar.set_description(printstr)

        if steps % config["distinct_chkpt_interval"] == 0:
            g_path = "%s/g_%d" % (chkpt_dir, steps)
            do_path = "%s/do_%d" % (chkpt_dir, steps)
            save_chkpts(g_path, do_path)

        if steps % config["val_interval"] == 0:
            g_path = "%s/g_latest" % (chkpt_dir)
            do_path = "%s/do_latest" % (chkpt_dir)
            save_chkpts(g_path, do_path)
            validate(steps)

        if steps == config["max_steps"]:
            continue_next_epoch = False
            break

    if not continue_next_epoch:
        break
