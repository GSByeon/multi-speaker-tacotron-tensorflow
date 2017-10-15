import os 
import matplotlib
from jamo import h2j, j2hcj

matplotlib.use('Agg')
matplotlib.rc('font', family="NanumBarunGothic")
import matplotlib.pyplot as plt

from text import PAD, EOS
from utils import add_postfix
from text.korean import normalize

def plot(alignment, info, text):
    char_len, audio_len = alignment.shape # 145, 200

    fig, ax = plt.subplots(figsize=(char_len/5, 5))
    im = ax.imshow(
            alignment.T,
            aspect='auto',
            origin='lower',
            interpolation='none')

    xlabel = 'Encoder timestep'
    ylabel = 'Decoder timestep'

    if info is not None:
        xlabel += '\n{}'.format(info)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if text:
        jamo_text = j2hcj(h2j(normalize(text)))
        pad = [PAD] * (char_len - len(jamo_text) - 1)

        plt.xticks(range(char_len),
                [tok for tok in jamo_text] + [EOS] + pad)

    if text is not None:
        while True:
            if text[-1] in [EOS, PAD]:
                text = text[:-1]
            else:
                break
        plt.title(text)

    plt.tight_layout()

def plot_alignment(
        alignment, path, info=None, text=None):

    if text:
        tmp_alignment = alignment[:len(h2j(text)) + 2]

        plot(tmp_alignment, info, text)
        plt.savefig(path, format='png')
    else:
        plot(alignment, info, text)
        plt.savefig(path, format='png')

    print(" [*] Plot saved: {}".format(path))
