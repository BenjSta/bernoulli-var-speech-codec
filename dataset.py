import numpy as np
import soundfile
import scipy.signal
from torch.utils import data
import torch
import os
import glob

def load_paths(dns4_root_datasets_fullband, vctk_txt_root):
    VOCALSET_PATH = os.path.join(dns4_root_datasets_fullband, "clean_fullband/VocalSet_48kHz_mono")

    VCTK_VAL_PATHS = [
        os.path.join(dns4_root_datasets_fullband, "clean_fullband/vctk_wav48_silence_trimmed/" + spk)
        for spk in ["p225", "p226", "p227", "p228"]
    ]

    VCTK_TEST_PATHS = [
        os.path.join(dns4_root_datasets_fullband, "clean_fullband/vctk_wav48_silence_trimmed/" + spk)
        for spk in ["p229", "p230", "p232", "p237"]
    ]

    training_filelist = glob.glob(
        os.path.join(dns4_root_datasets_fullband, "clean_fullband/**/*.wav"), recursive=True
    )
    training_filelist = [
        p
        for p in training_filelist
        if not VOCALSET_PATH in p
        or any(v in p for v in VCTK_VAL_PATHS)
        or any(v in p for v in VCTK_TEST_PATHS)
    ]

    np.random.seed(1)
    validation_filelist = []
    for p in VCTK_VAL_PATHS:
        validation_filelist += sorted(glob.glob(os.path.join(p, "*_mic1.wav"), recursive=True))
    np.random.seed()

    val_texts = []
    for v in validation_filelist:
        (dir, file) = os.path.split(v)
        (_, speakerdir) = os.path.split(dir)
        textfile = os.path.join(vctk_txt_root, speakerdir, file[:-9] + ".txt")
        with open(textfile, "r") as f:
            text = f.read()
            val_texts.append(text[:-1])

    np.random.seed(1)
    test_filelist = []
    for p in VCTK_TEST_PATHS:
        test_filelist += sorted(glob.glob(os.path.join(p, "*_mic1.wav"), recursive=True))
    np.random.seed()

    test_texts = []
    for v in test_filelist:
        (dir, file) = os.path.split(v)
        (_, speakerdir) = os.path.split(dir)
        textfile = os.path.join(vctk_txt_root, speakerdir, file[:-9] + ".txt")
        with open(textfile, "r") as f:
            text = f.read()
            test_texts.append(text[:-1])


    return {
        "clean": [training_filelist, validation_filelist, test_filelist],
        "txt": [None, val_texts, test_texts],
    }




def load_speech_sample(
                speech_filepath, fs,
                duration):
  
    if duration == None:
        s, fs_speech = soundfile.read(speech_filepath, dtype='float32')
        duration = s.shape[0] / fs_speech
        target_len = int(np.floor(duration * fs))
    else:
        target_len = int(np.floor(fs * duration)) 
        speech_info = soundfile.info(speech_filepath)
        speech_len = speech_info.frames

        target_len_speech = int(np.floor(speech_info.samplerate * duration))
        speech_start_index = np.random.randint(np.maximum(speech_len - target_len_speech, 0) + 1)
        
        s, fs_speech = soundfile.read(speech_filepath, frames=target_len_speech, 
                                        start=speech_start_index, dtype='float32')
    
    s = scipy.signal.resample_poly(s, fs, fs_speech)
    
    if s.shape[0] < target_len:
        pad_len = target_len - s.shape[0]
        pad_before = np.random.randint(pad_len)
        pad_after = pad_len - pad_before
        s = np.pad(s, (pad_before, pad_after))
    else:
        s = 10 ** (-10/20) * s[:target_len]

    return s.astype('float32')
            
class SpeechDataset(data.Dataset):
    def __init__(self,
                 speech_filepath_list,
                 duration,
                 fs):
        
        self.speech_filepath_list = speech_filepath_list
        self.duration = duration
        self.fs = fs

        self.length = len(speech_filepath_list)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        speech_filepath = self.speech_filepath_list[item]

        s = load_speech_sample(speech_filepath, self.fs, self.duration)
        return (s,)

