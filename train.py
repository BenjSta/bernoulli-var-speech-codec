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
from metrics import compute_dnsmos, compute_pesq, compute_mean_wacc, compute_mcd, compute_estimated_metrics, compute_visqol

from third_party.BigVGAN.models import (
    BigVGAN
)

from third_party.BigVGAN.utils import load_checkpoint
from bvrnn import BVRNN
from dataset import SpeechDataset, load_paths
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


# LOAD CONFIG
config = toml.load("configs_coding/config_16bit.toml")
vocoder_config_attr_dict = AttrDict(config['vocoder_config'])
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

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

np.random.seed(1)
scaler_mel_y = StandardScaler()
for i in tqdm.tqdm(np.random.choice(len(clean_train), 600, replace=False)):
    y = trainset.__getitem__(i)[0]
    s_mel_spectrogram = mel_spectrogram(torch.from_numpy(y[None, :]), **mel_spec_config)
    scaler_mel_y.partial_fit(s_mel_spectrogram[0, ...].T)
np.random.seed()

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

)


# load a vocoder for waveform generation
generator = BigVGAN(vocoder_config_attr_dict).to(device)
print("Generator params: {}".format(sum(p.numel()
      for p in generator.parameters())))

state_dict_g = load_checkpoint(config['vocoder_checkpoint'], device)

generator.load_state_dict(state_dict_g["generator"])
generator.eval()
print(generator)

bvrnn = BVRNN(config["num_mels"], config["h_dim"], config["z_dim"],
            [scaler_mel_y.mean_, np.sqrt(scaler_mel_y.var_)], config["log_sigma_init"]).to('cuda')
script_bvrnn = torch.jit.script(bvrnn)

chkpt_dir = os.path.join(chkpt_log_dirs['chkpt_log_dir'], config["train_name"], 'checkpoints')
os.makedirs(chkpt_dir, exist_ok=True)

optim = torch.optim.AdamW(
    bvrnn.parameters(),
    config['learning_rate'],
    betas=[config['adam_b1'], config['adam_b2']],
)

steps = 0
if config["resume"]:
    state_dict_vrnn = load_checkpoint(
        os.path.join(chkpt_dir, "latest"), device)
    bvrnn.load_state_dict(state_dict_vrnn["vrnn"])
    optim.load_state_dict(state_dict_vrnn["optim"])
    steps = state_dict_vrnn["steps"]


sched = torch.optim.lr_scheduler.ExponentialLR(optim, config['lr_decay'], last_epoch=steps-1)


def constr(input_res):
    return {
        "x": torch.ones((1, 80, input_res[0])).to(device),
        "length": 1 * config["fs"],
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

def constr(input_res):
    return {
        "y": torch.ones((1, input_res[0], 80)).to(device),
        "p_use_gen": 1.0, "greedy": True
    }
macs, params = get_model_complexity_info(
    bvrnn,
    (1 * 86,),
    input_constructor=constr,
    as_strings=False,
    print_per_layer_stat=True,
    verbose=True,
)
print("Computational complexity of VRNN model: %g" % macs)
print("Number of parameters in VRNN model: %g" % params)

validation_dataset = SpeechDataset(clean_val, None, 48000)

val_dataloader = DataLoader(
    validation_dataset,
    1,
    False,
    None,
    None,
    0,
)

log_dir = os.path.join(chkpt_log_dirs['chkpt_log_dir'], config['train_name'])
os.makedirs(log_dir, exist_ok=True)

sw = SummaryWriter(log_dir)

def validate(step):
    bvrnn.eval()
    script_bvrnn.eval()

    np.random.seed(1)

    y_all = []
    reconst_all = []
    elbo_all = []
    kld_all = []
    nll_all = []

    for batch in tqdm.tqdm(val_dataloader):
        with torch.no_grad():
            y = batch[0]
            y_all.append(y[0, :].detach().cpu().numpy())
            y = torch.from_numpy(
                scipy.signal.resample_poly(
                    y.detach().cpu().numpy(), config["fs"], 48000, axis=1
                )
            )
            y = y.to(device, non_blocking=True)
            valid_length = y.shape[1] // config["hopsize"] * config["hopsize"]
            y = y[:, :valid_length]

            y_mel = mel_spectrogram(y, **mel_spec_config)
            y_mel_reconst, kld = bvrnn(y_mel.permute(0, 2, 1), 1.0, True)
            y_mel_reconst = y_mel_reconst.permute(0, 2, 1)
            mae = torch.mean(torch.abs(y_mel - y_mel_reconst))

            nll = y_mel_reconst.shape[1] * bvrnn.log_sigma + \
                1/(torch.exp(bvrnn.log_sigma)) * y_mel_reconst.shape[1] * mae
            elbo = nll + kld
            elbo_all.append((elbo).detach().cpu().numpy())
            nll_all.append((nll).detach().cpu().numpy())
            kld_all.append((kld).detach().cpu().numpy())
            y_g_hat = generator(y_mel_reconst, y.shape[1])
            
            reconst_all.append(scipy.signal.resample_poly(y_g_hat[0, 0, :].detach().cpu().numpy(), 48000, config['fs'])) 

    clean_all = y_all
    sigs_method = reconst_all 
    lengths_all = np.array([y.shape[0] for y in y_all])

    pesq = compute_pesq(clean_all, sigs_method, 48000)
    stoi_est, pesq_est, sisdr_est, mos = compute_estimated_metrics(clean_all, sigs_method, 48000) 
    ovr, sig, bak = compute_dnsmos(sigs_method, 48000)            
    mcd = compute_mcd(clean_all, sigs_method, 48000)
    visqol = compute_visqol( executables['visqol_base'],  executables['visqol_bin'], clean_all, sigs_method, 48000)

    mean_elbo = np.mean(lengths_all * np.array(elbo_all)) / np.mean(lengths_all)
    mean_kld = np.mean(lengths_all * np.array(kld_all)) / np.mean(lengths_all)
    mean_nll = np.mean(lengths_all * np.array(nll_all)) / np.mean(lengths_all)
    mean_pesq = np.mean(lengths_all * np.array(pesq)) / np.mean(lengths_all)
    mean_sig = np.mean(lengths_all * np.array(sig)) / np.mean(lengths_all)
    mean_ovr = np.mean(lengths_all * np.array(ovr)) / np.mean(lengths_all)
    mean_bak = np.mean(lengths_all * np.array(bak) / np.mean(lengths_all))
    mean_mcd = np.mean(lengths_all * np.array(mcd)) / np.mean(lengths_all)
    mean_stoi_est = np.mean(lengths_all * np.array(stoi_est) / np.mean(lengths_all))
    mean_pesq_est = np.mean(lengths_all * np.array(pesq_est) / np.mean(lengths_all))
    mean_sisdr_est = np.mean(lengths_all * np.array(sisdr_est) / np.mean(lengths_all))
    mean_mos_est = np.mean(lengths_all * np.array(mos) / np.mean(lengths_all))
    mean_visqol = np.mean(lengths_all * np.array(visqol) / np.mean(lengths_all))

    mean_wacc = compute_mean_wacc(sigs_method, txt_val, 48000, 'cuda')

    sw.add_scalar('PESQ', mean_pesq, 0)
    sw.add_scalar('SIG', mean_sig, 0)
    sw.add_scalar('OVR', mean_ovr, 0)
    sw.add_scalar('BAK', mean_bak, 0)
    sw.add_scalar('MCD', mean_mcd, 0)
    sw.add_scalar('WAcc', mean_wacc, 0)
    sw.add_scalar('STOI-est.', mean_stoi_est, 0)
    sw.add_scalar('PESQ-est.', mean_pesq_est, 0)
    sw.add_scalar('SI-SDR-est.', mean_sisdr_est, 0)
    sw.add_scalar('MOS-est', mean_mos_est, 0)
    sw.add_scalar('Visqol', mean_visqol, 0)
    sw.add_scalar("KLD", mean_kld, step)
    sw.add_scalar("NLL", mean_nll, step)
    sw.add_scalar("ELBO", mean_elbo, step)
    sw.add_scalar("Sigma", torch.exp(bvrnn.log_sigma), step)

    for i in val_tensorboard_examples:
        sw.add_audio(
            "%d" % i,
            torch.from_numpy(reconst_all[i]),
            global_step=step,
            sample_rate=48000,
        )
    sw.flush()
    sw.close()
    bvrnn.train()
    script_bvrnn.train()
    np.random.seed()


if config["validate_only"]:
    validate(steps)
    exit()

# main training loop
bvrnn.train()
script_bvrnn.train()


while True:
    continue_next_epoch = True
    pbar = tqdm.tqdm(train_dataloader)
    for batch in pbar:
        y = batch[0]
        y = y.to(device, non_blocking=True)
        valid_length = y.shape[1] // config["hopsize"] * config["hopsize"]
        y = y[:, :valid_length]

        optim.zero_grad()
        y_mel = mel_spectrogram(y, **mel_spec_config)
        p_use_gen = 1.0-(0.01**(steps/config['teacher_force_step_1perc']))
        if steps > config['teacher_force_step_1perc']:
            p_use_gen = 1.0
        y_mel_reconst, kld = script_bvrnn(y_mel.permute(0, 2, 1), p_use_gen, False)
        y_mel_reconst = y_mel_reconst.permute(0, 2, 1)

        mae = torch.mean(torch.abs(y_mel - y_mel_reconst))

        nll = y_mel_reconst.shape[1] * bvrnn.log_sigma + \
            1/(torch.exp(bvrnn.log_sigma)) * y_mel_reconst.shape[1] * mae
        elbo = nll + kld
        elbo.backward()
        optim.step()
        sched.step()

        steps += 1
        # # STDOUT logging
        printstr = (
            "Steps : %d, KL loss : %g, NLL : %g, ELBO : %g, LR: %g, p: %g, MAE: %g"%
            (
                steps,
                kld.detach().cpu().numpy(),
                nll.detach().cpu().numpy(),
                elbo.detach().cpu().numpy(),
                sched.get_last_lr()[0],
                p_use_gen, mae
            )
            )
        pbar.set_description(printstr)

        if steps % config["distinct_chkpt_interval"] == 0:
            chkpt_path = "%s/%d" % (chkpt_dir, steps)
            torch.save({
                "vrnn": bvrnn.state_dict(),
                "optim": optim.state_dict(),
                "steps": steps,
                }, chkpt_path)

        if steps % config["val_interval"] == 0:
            chkpt_path = "%s/latest" % (chkpt_dir)
            torch.save({
                "vrnn": bvrnn.state_dict(),
                "optim": optim.state_dict(),
                "steps": steps,
                }, chkpt_path)
            validate(steps)

        if steps == config["max_steps"]:
            continue_next_epoch = False
            break

    if not continue_next_epoch:
        break
