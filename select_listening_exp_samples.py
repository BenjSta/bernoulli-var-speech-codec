import os
import numpy as np
import soundfile
import glob
from dataset import load_paths
import toml
import scipy.signal

MINLEN = 2
MAXLEN = 4
CUTSTART_VCTK = 0.6
CUTSTART_TSP = 0.0
CUTEND = 0.05

prefixes_names = [
    ('clean', 'ref'),
    ('audiodec8', 'audiodec_8'),
    ('encodec6', 'encodec_6'),
    ('encodec1_5', 'encodec_15'),
    ('lyra6', 'lyra_6'),
    ('lyra3_2', 'lyra_32'),
    ('variable16_ft', 'prop_13_16k'),
    ('variable16_ft', 'prop_13'),
    ('variable64_ft', 'prop_55_16k'),
    ('variable64_ft', 'prop_55'),
]

for i in range(1, 17):
    os.makedirs('stim/stim_%s' % str(i).zfill(2), exist_ok=True)

pathconfig = toml.load("data_directories.toml")
paths = load_paths(pathconfig["DNS4_root"], pathconfig["VCTK_txt_root"])
clean_train, clean_val, clean_test = paths["clean"]

vctk_outp = os.listdir('results_vctk')
vctk_outp = [x for x in vctk_outp if x.endswith(
    '.flac') and x.startswith('clean')]
vctk_outp = sorted(vctk_outp)
print(len(vctk_outp))

np.random.seed(42)

for spkoffset, speaker in enumerate(["p229", "p230", "p232", "p237"]):
    speaker_ind = [i for i, t in enumerate(clean_test) if speaker in t]
    vctk_outp_spk = ['clean_%d.flac' % i for i in speaker_ind]
    print(len(vctk_outp_spk))

    speaker_ind_long_enough = []
    for i, v in zip(speaker_ind, vctk_outp_spk):
        info = soundfile.info('results_vctk/' + v)
        if (info.duration >= MINLEN + CUTSTART_VCTK + CUTEND) and (info.duration <= MAXLEN + CUTSTART_VCTK + CUTEND):
            speaker_ind_long_enough.append(i)
        
    print(len(speaker_ind_long_enough))

    speaker_ind_long_enough = np.random.choice(speaker_ind_long_enough, 2, replace=False)
    for i, v in enumerate(speaker_ind_long_enough):
        for pr, name in prefixes_names:
            x, fs = soundfile.read('results_vctk/' + pr + '_%d.flac' % v)
            

            # startind = np.random.randint(int(0.025 * fs), len(x) - int(fs * (LEN + 0.075)))
            # endind = startind + int(fs * (LEN + 0.05))
            # x = x[startind:endind]
            if name == 'ref':
                endindex = x.shape[0]-int(fs * CUTEND)

            x = x[int(fs * CUTSTART_VCTK):endindex]
            startramp = np.linspace(0, 1, int(0.025 * fs))
            x[:int(0.025 * fs)] = startramp * x[:int(0.025 * fs)]
            endramp = np.linspace(1, 0, int(0.025 * fs))
            x[-int(0.025 * fs):] = endramp * x[-int(0.025 * fs):]
            
            x24k = scipy.signal.resample_poly(x, 24000, fs)
            l = x24k.shape[0]
            if '16k' in name:
                x16k = scipy.signal.resample_poly(x24k, 16000, 24000)
                x24k = scipy.signal.resample_poly(x16k, 24000, 16000)[:l]

            soundfile.write('stim/stim_%s/%s.wav' % (str(spkoffset * 2 + i+1).zfill(2), name), np.stack(2 * [x24k], 1), 24000)


tsp_outp = os.listdir('results_tsp')

tsp_outp = [x for x in tsp_outp if x.endswith(
    '.flac') and x.startswith('clean')]
tsp_outp = sorted(tsp_outp)

print(len(tsp_outp))


tsp_long_enough_indices = []
sumdur_tsp = 0
for i, v in enumerate(tsp_outp):
    info = soundfile.info('results_tsp/' + v)
    sumdur_tsp += info.duration
    if (info.duration >= MINLEN + CUTSTART_TSP + CUTEND) and (info.duration <= MAXLEN + CUTSTART_TSP + CUTEND):
        tsp_long_enough_indices.append(i)

print(len(tsp_long_enough_indices))

tsp_filelist = sorted(glob.glob('/media/DATAslow/shared/stahl/mtedx_iwslt2021/48k/**/*.wav'))
female_indices = [i for i, t in enumerate(tsp_filelist) if t.split('/')[-2].startswith('F') and i in tsp_long_enough_indices]
male_indices = [i for i, t in enumerate(tsp_filelist) if t.split('/')[-2].startswith('M') and i in tsp_long_enough_indices]

female_indices = np.random.choice(female_indices, 4, replace=False)
for i, v in enumerate(female_indices):
    for pr, name in prefixes_names:
        x, fs = soundfile.read('results_tsp/' + pr + '_%d.flac' % v)
        # startind = np.random.randint(int(0.025 * fs), len(x) - int(fs * (LEN + 0.075)))
        # endind = startind + int(fs * (LEN + 0.05))
        # x = x[startind:endind]
        if name == 'ref':
            endindex = x.shape[0]-int(fs * CUTEND)
        x = x[int(fs * CUTSTART_TSP):endindex]
        startramp = np.linspace(0, 1, int(0.025 * fs))
        x[:int(0.025 * fs)] = startramp * x[:int(0.025 * fs)]
        endramp = np.linspace(1, 0, int(0.025 * fs))
        x[-int(0.025 * fs):] = endramp * x[-int(0.025 * fs):]

        x24k = scipy.signal.resample_poly(x, 24000, fs)
        l = x24k.shape[0]
        if '16k' in name:
            x16k = scipy.signal.resample_poly(x24k, 16000, 24000)
            x24k = scipy.signal.resample_poly(x16k, 24000, 16000)[:l]

        soundfile.write('stim/stim_%s/%s.wav' % (str(i + 9).zfill(2), name), np.stack(2 * [x24k], 1), 24000)


male_indices = np.random.choice(male_indices, 4, replace=False)
for i, v in enumerate(male_indices):
    for pr, name in prefixes_names:
        x, fs = soundfile.read('results_tsp/' + pr + '_%d.flac' % v)
        # startind = np.random.randint(int(0.025 * fs), len(x) - int(fs * (LEN + 0.075)))
        # endind = startind + int(fs * (LEN + 0.05))
        # x = x[startind:endind]
        if name == 'ref':
            endindex = x.shape[0]-int(fs * CUTEND)
        x = x[int(fs * CUTSTART_TSP):endindex]
        startramp = np.linspace(0, 1, int(0.025 * fs))
        x[:int(0.025 * fs)] = startramp * x[:int(0.025 * fs)]
        endramp = np.linspace(1, 0, int(0.025 * fs))
        x[-int(0.025 * fs):] = endramp * x[-int(0.025 * fs):]

        x24k = scipy.signal.resample_poly(x, 24000, fs)
        l = x24k.shape[0]
        if '16k' in name:
            x16k = scipy.signal.resample_poly(x24k, 16000, 24000)
            x24k = scipy.signal.resample_poly(x16k, 24000, 16000)[:l]

        print(x24k.shape[0], name)

        soundfile.write('stim/stim_%s/%s.wav' % (str(i + 13).zfill(2), name), np.stack(2 * [x24k], 1), 24000)

