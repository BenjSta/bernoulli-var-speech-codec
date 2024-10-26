from dataset import SpeechDataset, load_paths
import toml
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
# _, txt_val, _ = paths["txt"]


trainset = SpeechDataset(
    clean_train,
    duration=4.0,
    fs=48000,
)
import soundfile

import tqdm
dur_total = 0
for c in  tqdm.tqdm(clean_train):
    info =soundfile.info(c)
    #print(info.subtype_info)
    dur = info.duration
    dur_total += dur

print("Total duration of training set: ", dur_total / 3600)


import numpy as np
samples = np.random.choice(len(trainset), 10000, replace=False)


has10dBpad_all = []
for i in tqdm.tqdm(samples):
    _, has10dBpad = trainset.__getitem__(i)
    has10dBpad_all.append(has10dBpad)

print("Fraction of samples with 10dB pad: ", np.mean(has10dBpad_all))

