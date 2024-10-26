import os
import pandas as pd
import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator


pesq_vals = []
wacc_vals = []
sig_vals = []

LOGDIR = "/media/DATA/shared/stahl/bernoulli-var-speech-codec/vocoder_fine_tune/logs/snake_causal_doubleseg_BVRNN"

steps = np.arange(2500000, 3500001, 20000)

best_metric = 0
for step in steps:
    nisqa = None
    pesq = None
    visqol = None
    for f in os.listdir(LOGDIR):
        if not f.endswith('.csv'):
            for e in summary_iterator(os.path.join(LOGDIR, f)):
                if e.step == step:
                    #if e.content_type == "scalar":
                    for v in e.summary.value:
                        if v.tag == 'NISQA':
                            nisqa = v.simple_value
                        elif v.tag == 'PESQ':
                            pesq = v.simple_value
                        elif v.tag == 'ViSQOL':
                            visqol = v.simple_value
    
    metric = 0.5 * nisqa + 0.5 * visqol
    if metric > best_metric:
        best_metric = metric
        best_step = step

print(best_step)