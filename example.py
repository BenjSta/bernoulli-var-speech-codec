from bvrnn_codec_model import BVRNNCodecModel
import numpy as np
import torch
import soundfile
import scipy.signal

# instantiate the codec
codec_model = BVRNNCodecModel()
codec_model.eval()

# load a speech file, use first channel:
speech, fs_speech = soundfile.read('./mushra_results_dataset/audio/stim_01/ref.wav', always_2d=True)
speech = speech[:, 0]

# resample to 22050Hz and normalize
speech = scipy.signal.resample_poly(speech, 22050, fs_speech)
speech = speech / np.max(np.abs(speech))

## Example 1: Encode with 3kbps and decode in a single step
decoded = codec_model(torch.from_numpy(speech).float()[None, :], 3000)[0, :].detach().cpu().numpy()
soundfile.write('./stim_01_decoded.wav', decoded, 22050)

## Example 2: Encoding and decoding in two steps
length = speech.shape[0]
codes = codec_model.encode(torch.from_numpy(speech).float()[None, :], 3000)
decoded = codec_model.decode(codes, length)[0, :].detach().cpu().numpy()
soundfile.write('./stim_01_decoded2.wav', decoded, 22050)

