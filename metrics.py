import scipy.signal
import numpy as np
from werpy import wer as compute_wer, normalize as norm_text
from third_party.dnsmos_local import ComputeScore
from pesq import pesq
import tqdm
from pymcd.mcd import Calculate_MCD
import tempfile
import soundfile
import torch
import whisper
import os
import subprocess

def compute_mcd(list_of_refs, list_of_signals, fs):
    mcd_toolbox = Calculate_MCD(MCD_mode="plain")
    mcdvals = []
    pbar = tqdm.tqdm(zip(list_of_refs, list_of_signals))
    pbar.set_description("Computing MCD...")
    for r, s in pbar:
        with tempfile.NamedTemporaryFile(suffix='.wav') as reffile, tempfile.NamedTemporaryFile(suffix='.wav') as sigfile:
            refname = reffile.name
            signame = sigfile.name
        
            soundfile.write(refname, r, fs)
            soundfile.write(sigfile, s, fs)
        
            mcdvals.append(mcd_toolbox.calculate_mcd(refname, signame))

    return np.array(mcdvals)


def compute_pesq(list_of_refs, list_of_signals, fs):
    pesqvals = []
    pbar = tqdm.tqdm(zip(list_of_refs, list_of_signals))
    pbar.set_description("Computing PESQ...")
    for r, s in pbar:
        r = scipy.signal.resample_poly(r, 16000, fs)
        s = scipy.signal.resample_poly(s, 16000, fs)
        try:
            pesqval = pesq(16000, r, s, "wb")
        except:
            pesqval = np.nan

        pesqvals.append(pesqval)

    return np.array(pesqvals)


def compute_dnsmos(list_of_signals, fs):
    dnsmos_obj = ComputeScore(
        "third_party/DNSMOS/sig_bak_ovr.onnx", "third_party/DNSMOS/model_v8.onnx"
    )

    dnsmos_sig = []
    dnsmos_bak = []
    dnsmos_ovrl = []
    lengths = []

    pbar = tqdm.tqdm(list_of_signals)
    pbar.set_description("Computing DNSMOS...")

    for s in pbar:
        d = dnsmos_obj(scipy.signal.resample_poly(s / np.max(np.abs(s)), 16000, fs))
        dnsmos_ovrl.append(d["OVRL"])
        dnsmos_bak.append(d["SIG"])
        dnsmos_sig.append(d["BAK"])
        lengths.append(s.shape[0])

    return (np.array(dnsmos_ovrl), np.array(dnsmos_sig), np.array(dnsmos_bak))


def compute_mean_wacc(list_of_signals, list_of_texts, fs, device):
    list_of_transcripts = []
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    asr_model = whisper.load_model("medium.en", device = device)
    pbar = tqdm.tqdm(list_of_signals)
    pbar.set_description("Computing Wacc...")

    for s in pbar:
        list_of_transcripts.append(
            asr_model.transcribe(
                scipy.signal.resample_poly(s / np.max(np.abs(s)), 16000, fs)
            )["text"]
        )
    norm_list_of_transcripts = [' ' if i =='' else i for i in norm_text(list_of_transcripts)]

    return 1 - compute_wer(norm_text(list_of_texts), norm_list_of_transcripts)

def visqol(ref, sig, fs, visqol_base, visqol_executable):
    with tempfile.NamedTemporaryFile(suffix='.wav', mode='r+') as reffile, tempfile.NamedTemporaryFile(suffix='.wav', mode='r+') as sigfile:
        soundfile.write(
            reffile.name, scipy.signal.resample_poly(ref, 48000, fs), 48000)
        soundfile.write(
            sigfile.name, scipy.signal.resample_poly(sig, 48000, fs), 48000)
        cwd = os.getcwd()
        os.chdir(visqol_base)
        retval = subprocess.check_output('%s --reference_file "%s" --degraded_file "%s" --use_speech_mode' % (
            visqol_executable, reffile.name, sigfile.name), shell=True)
        
    return y.astype('float32')
 
