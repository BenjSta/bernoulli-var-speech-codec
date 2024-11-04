import toml
import os

# LOAD CONFIG
config = toml.load("configs_finetuning/config_vocoder_causal_snake_doubleseg_bvrnn.toml")
os.environ["CUDA_VISIBLE_DEVICES"] = config["cuda_visible_devices"]

from third_party.BigVGAN.env import AttrDict
import platform
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import scipy.signal
import tqdm
import itertools
from ptflops import get_model_complexity_info
import torch
from third_party.BigVGAN.meldataset import mel_spectrogram
from bvrnn import BVRNN

config_attr_dict = AttrDict(config)

from third_party.BigVGAN.models import (
    BigVGAN,
    MultiPeriodDiscriminator,
    MultiResolutionDiscriminator,
    feature_loss,
    generator_loss,
    discriminator_loss,
)

from third_party.BigVGAN.utils import load_checkpoint, save_checkpoint
from dataset import SpeechDataset, load_paths
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

torch.backends.cudnn.benchmark = True



if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

chkpt_log_config = toml.load("chkpt_log_dirs.toml")
pathconfig = toml.load("data_directories.toml")
paths = load_paths(pathconfig["DNS4_root"], pathconfig["VCTK_txt_root"])
clean_train, clean_val, _ = paths["clean"]
_, txt_val, _ = paths["txt"]

try:
    executables = toml.load("executable_paths.toml")
except:
    raise RuntimeError(
        'Error loading executable_paths.toml, please create it based on the corresponding default file')


np.random.seed(1)
val_tensorboard_examples = np.random.choice(len(clean_val), 15, replace=False)
np.random.seed()


trainset = SpeechDataset(
    clean_train, duration=config["train_seq_duration"], fs=config["fs"])

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
    chkpt_log_config['chkpt_log_dir'], "vocoder_fine_tune", "checkpoints", config["train_name"])
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



def getBitrate():
    # if np.random.rand() <0.5:
    #     Bitrate = np.floor(np.random.uniform(1,25))
    # else: 
    #     Bitrate = np.floor(np.exp(np.random.uniform(np.log(25),np.log(65))))
    return np.floor(np.random.uniform(1,65))

def getNumQuantStages():
    # if np.random.rand() <0.5:
    #     Bitrate = np.floor(np.random.uniform(1,25))
    # else: 
    #     Bitrate = np.floor(np.exp(np.random.uniform(np.log(25),np.log(65))))
    return np.floor(np.random.uniform(1,25))

state_dict_g = load_checkpoint(config['vocoder_chkpt_g'], device)
state_dict_do = load_checkpoint(config['vocoder_chkpt_do'], device)
generator.load_state_dict(state_dict_g["generator"])
optim_g.load_state_dict(state_dict_do["optim_g"])
optim_d.load_state_dict(state_dict_do["optim_d"])
mpd.load_state_dict(state_dict_do["mpd"])
mrd.load_state_dict(state_dict_do["mrd"])
steps = state_dict_do["steps"]

if config["coder_type"] == 'bvrnn':
    bvrnn = BVRNN(config["num_mels"], config["h_dim"], config["z_dim"],
                [np.zeros(80), np.ones(80)], config["log_sigma_init"], config["var_bit"]).to('cuda')
    script_bvrnn = torch.jit.script(bvrnn)
    state_dict_vrnn = load_checkpoint(config['bvrnn_chkpt'], device)
    bvrnn.load_state_dict(state_dict_vrnn["vrnn"])
    rand_bitrate_fun = getBitrate
elif config["coder_type"] == 'vq':
    import pickle
    with open('vq_coeffs/quantizers2.pkl', 'rb') as f:
        quantizers = pickle.load(f)
    with open('vq_coeffs/mu.pkl', 'rb') as f:
        mu = pickle.load(f)
    with open('vq_coeffs/vh.pkl', 'rb') as f:
        vh = pickle.load(f)

    
    def quantizer_encode_decode_batched(mel_spec, layers):
        # mel spec shape: (batch_size, num_time_frames, num_mels)
        # layers: (batch_size, num_time_frames)
        num_time_frames = mel_spec.shape[1]
        num_supervectors = num_time_frames // 3
        valid_length = num_supervectors * 3
        mel_spec = mel_spec[:, :valid_length, :]
        layers = layers[:, :valid_length]
        mel_spec = mel_spec.reshape(mel_spec.shape[0], num_supervectors, 3 * mel_spec.shape[2])
        layers = layers.reshape(layers.shape[0], num_supervectors, 3)
        layers = np.min(layers, axis=2)

        batch_size = mel_spec.shape[0]

        mel_spec = mel_spec.reshape(-1, mel_spec.shape[2])
        layers = layers.reshape(-1)

        mel_spec = mel_spec - mu
        residual = mel_spec @ (vh.T)
        
        reconst = np.zeros_like(mel_spec)

        for i, q in enumerate(quantizers):
            quantized = q.cluster_centers_[q.predict(residual), :]
            residual = residual - quantized
            # only add up to number of layers for a certain sequence/time frame
            reconst += (i < layers).astype('float32')[:, None] * quantized
        
        reconst = reconst @ vh
        reconst = reconst + mu
        reconst = reconst.reshape(batch_size, num_supervectors, -1)
        reconst = reconst.reshape(reconst.shape[0], valid_length, -1)
        return reconst
    rand_bitrate_fun = getNumQuantStages

else:
    raise ValueError('Invalid coder type')

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

validation_dataset = SpeechDataset(
    clean_val, duration=None, fs=48000)

val_dataloader = DataLoader(
    validation_dataset,
    1,
    False,
    None,
    None,
    0,
)

log_dir = os.path.join(chkpt_log_config["chkpt_log_dir"], "vocoder_fine_tune", "logs", config["train_name"])
os.makedirs(log_dir, exist_ok=True)

sw_proposed = SummaryWriter(log_dir)

mel_spec_config = {'n_fft': config["winsize"],
                   'num_mels': config["num_mels"],
                   'sampling_rate': config["fs"],
                   'hop_size': config["hopsize"],
                   'win_size': config["winsize"],
                   'fmin': config["fmin"],
                   'fmax': config["fmax"],
                   'padding_left': config["mel_pad_left"]}

mel_spec_config_loss = {'n_fft': config["winsize"],
                        'num_mels': config["msl_num_mels"],
                        'sampling_rate': config["fs"],
                        'hop_size': config["hopsize"],
                        'win_size': config["winsize"],
                        'fmin': config["msl_fmin"],
                        'fmax': config["msl_fmax"],
                        'padding_left': config["mel_pad_left"]}



def validate(step):
    generator.eval()
    mpd.eval()
    mrd.eval()

    np.random.seed(1)

    y_all = []
    reconst_all = []
    bitrates = []

    for batch in tqdm.tqdm(val_dataloader):
        (y,) = batch
        y = torch.from_numpy(
            scipy.signal.resample_poly(
                y.detach().cpu().numpy(), config["fs"], 48000, axis=1
            )
        )
        y = y.to(device, non_blocking=True)
        valid_length = y.shape[1] // config["hopsize"] * config["hopsize"]
        y = y[:, :valid_length]

        y_mel = mel_spectrogram(y, **mel_spec_config)
        

        with torch.no_grad():
            varBitrateTens = np.Inf * torch.ones((y_mel.shape[0], y_mel.shape[2])).to(device)
        
            if config['var_bit']:
                for b in range(y_mel.shape[0]): 
                    change_index = np.random.randint(0, y_mel.shape[2])
                    time_tensor = torch.arange(0, y_mel.shape[2], 1).to(device)
                    if np.random.rand() < config['p_bitratechange']:
                        b1 =  rand_bitrate_fun()
                        b2 =  rand_bitrate_fun()
                        varBitrateTens[b,time_tensor <= change_index] = b1
                        varBitrateTens[b,time_tensor > change_index] = b2
                        bitrates.append((b1, b2))
                    else:
                        b1 =  rand_bitrate_fun()
                        varBitrateTens[b,:] = b1
                        bitrates.append((b1, ))

            if config["coder_type"] == 'bvrnn':
                y_mel_reconst, _ = script_bvrnn(y_mel.permute(0, 2, 1), 1.0, True, varBitrateTens)
            else:
                y_mel_reconst = torch.from_numpy(quantizer_encode_decode_batched(
                    y_mel.permute(0, 2, 1).cpu().numpy(), varBitrateTens.cpu().numpy()).astype(
                        'float32')).to(device)
                
            y_g_hat = generator(y_mel_reconst.permute(0, 2, 1), y.shape[1])


        reconst_all.append(y_g_hat[0, 0, :].detach().cpu().numpy())
        y_all.append(y[0, :].detach().cpu().numpy())

    
    lengths_all = np.array([y.shape[0] for y in y_all])
    
    for i in val_tensorboard_examples:
        b = bitrates[i]
        if len(b) == 1:
            bitrate_str = "varbit_%d" % b[0]
        elif len(b) == 2:
            bitrate_str = "varbit_%d->%d" % (b[0], b[1])
        sw_proposed.add_audio(
            "%d_%s" % (i, bitrate_str),
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

validate(steps)
if config["validate_only"]:
    exit()


# main training loop
generator.train()
mpd.train()
mrd.train()

while True:
    continue_next_epoch = True
    pbar = tqdm.tqdm(train_dataloader)
    for batch in pbar:
        (y,) = batch
        y = y.to(device, non_blocking=True)
        valid_length = y.shape[1] // config["hopsize"] * config["hopsize"]
        y = y[:, :valid_length]

        y_mel = mel_spectrogram(y, **mel_spec_config)

        varBitrateTens = np.Inf * torch.ones((y_mel.shape[0], y_mel.shape[2])).to(device)
        
        if config['var_bit']:
            for b in range(y_mel.shape[0]): 
                change_index = np.random.randint(0, y_mel.shape[2])
                time_tensor = torch.arange(0, y_mel.shape[2], 1).to(device)
                if np.random.rand() < config['p_bitratechange']:
                    varBitrateTens[b,time_tensor <= change_index] = rand_bitrate_fun()
                    varBitrateTens[b,time_tensor > change_index] = rand_bitrate_fun()
                else:
                    varBitrateTens[b,:] = rand_bitrate_fun()

        if config["coder_type"] == 'bvrnn':
            with torch.no_grad():
                y_mel_reconst, _ = script_bvrnn(y_mel.permute(0, 2, 1), 1.0, True, varBitrateTens)
        else:
            y_mel_reconst = torch.from_numpy(quantizer_encode_decode_batched(
                y_mel.permute(0, 2, 1).cpu().numpy(), varBitrateTens.cpu().numpy()).astype('float32')).to(device)

        NUM_CHUNKS = 5
        full_len = y_mel_reconst.shape[1]
        len_per_chunk = full_len // NUM_CHUNKS
        y_mel_reconst = y_mel_reconst[:, :len_per_chunk*5, :]
        y_mel_for_loss = mel_spectrogram(10**(10/20)*y, **mel_spec_config_loss)
        y_mel_for_loss = y_mel_for_loss[:, :, :len_per_chunk*5]
        y_mel_for_loss_chunks = y_mel_for_loss.reshape(y_mel_reconst.shape[0], -1, 5, len_per_chunk)

        y_mel_reconst_chunks = y_mel_reconst.reshape(y_mel_reconst.shape[0], 5, len_per_chunk, -1)
        y = y[:, :len_per_chunk*5*config["hopsize"]]
        y_chunks = y.reshape(y.shape[0], 5, len_per_chunk*config["hopsize"]) 

        for c in range(NUM_CHUNKS):
            y = y_chunks[:, c, :]
            y_mel_reconst = y_mel_reconst_chunks[:, c, :, :].permute(0, 2, 1)
            y_mel_for_loss = y_mel_for_loss_chunks[:, :, c, :]
            

            # multiply y and y_g_hat to scale up
            y_g_hat = 10**(10/20)*generator(y_mel_reconst, y.shape[1])
            y = 10**(10/20)*y

            y_g_hat_mel = mel_spectrogram(
                y_g_hat.squeeze(1), **mel_spec_config_loss)

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

    if not continue_next_epoch:
        break
