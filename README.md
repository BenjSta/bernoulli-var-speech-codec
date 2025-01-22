# bernoulli-var-speech-codec

This repository provides code for the approach described in

```
@ARTICLE{stahl2024codec,
  author={Stahl, Benjamin and Windtner, Simon and Sontacchi, Alois},
  journal={IEEE Access}, 
  title={A Bitrate-Scalable Variational Recurrent Mel-Spectrogram Coder for Real-Time Resynthesis-Based Speech Coding}, 
  year={2024},
  volume={12},
  number={},
  pages={159239-159251},
  keywords={Vocoders;Recurrent neural networks;Encoding;Convolutional codes;Decoding;Vectors;Speech coding;Training;Real-time systems;Data models;Speech codecs;recurrent neural networks;binary codes;generative adversarial networks;vocoders},
  doi={10.1109/ACCESS.2024.3482359}}
```

## Application
The proposed codec is used to code speech directly into binary codes in real time (total algorithmic latency: 34.8 ms). The model weights provided have been trained on **clean speech only**. Note that in order to use the model for **noisy speech**, retraining of the binary variational recurrent mel spectrogram coder as well as fine-tuning of the vocoder would be necessary.


## Dependencies
Tested with Python 3.11. Please have the following packages installed:

- torch
- numpy
- scipy
- librosa
- soundfile
- toml
- tqdm
- matplotlib

## Usage

Simply run the codec as follows:

```python
from bvrnn_codec_model import BVRNNCodecModel
import numpy as np
import torch
import soundfile
import scipy.signal

# instantiate the codec
codec_model = BVRNNCodecModel()
codec_model.eval()

# load a speech file, use first channel:
speech, fs_speech = soundfile.read('my_speech.wav', always_2d=True)
speech = speech[:, 0]

# resample to 22050Hz and normalize
speech = scipy.signal.resample_poly(speech, 22050, fs_speech)
speech = speech / np.max(np.abs(speech))

# encode with 3kbps and decode in a single step
decoded = codec_model(torch.from_numpy(speech).float()[None, :], 3000)[0, :].detach().cpu().numpy()

soundfile.write('my_speech_decoded.wav', decoded, 22050)
```

The encoding and decoding steps can also be done separately:
```python
length = speech.shape[0]
codes = codec_model.encode(torch.from_numpy(speech).float()[None, :], 3000)
decoded = codec_model.decode(codes, length)[0, :].detach().cpu().numpy()
```

## Dataset
We conducted a MUSHRA experiment to verify the quality of the proposed model.
We publish the input stimuli and raw MUSHRA ratings. The dataset contains 112 (excluding Reference and Anchor) rated test stimuli.

## License
The model weights and code are released under the Creative Commons Attribution-NonCommercial 4.0 International license. The accompanying dataset is released under the Creative Commons Attribution 4.0 International license.
 