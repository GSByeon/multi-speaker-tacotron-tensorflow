# Code based on https://github.com/keithito/tacotron/blob/master/datasets/ljspeech.py
import os
import re
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob
from functools import partial

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from hparams import hparams
from text import text_to_sequence
from utils import makedirs, remove_file, warning
from audio import load_audio, spectrogram, melspectrogram, frames_to_hours

def one(x=None):
    return 1

def build_from_path(config):
    warning("Sampling rate: {}".format(hparams.sample_rate))

    executor = ProcessPoolExecutor(max_workers=config.num_workers)
    futures = []
    index = 1

    base_dir = os.path.dirname(config.metadata_path)
    data_dir = os.path.join(base_dir, config.data_dirname)
    makedirs(data_dir)

    loss_coeff = defaultdict(one)
    if config.metadata_path.endswith("json"):
        with open(config.metadata_path) as f:
            content = f.read()
        info = json.loads(content)
    elif config.metadata_path.endswith("csv"):
        with open(config.metadata_path) as f:
            info = {}
            for line in f:
                path, text = line.strip().split('|')
                info[path] = text
    else:
        raise Exception(" [!] Unkown metadata format: {}".format(config.metadata_path))

    new_info = {}
    for path in info.keys():
        if not os.path.exists(path):
            new_path = os.path.join(base_dir, path)
            if not os.path.exists(new_path):
                print(" [!] Audio not found: {}".format([path, new_path]))
                continue
        else:
            new_path = path

        new_info[new_path] = info[path]

    info = new_info

    for path in info.keys():
        if type(info[path]) == list:
            if hparams.ignore_recognition_level == 1 and len(info[path]) == 1 or \
                    hparams.ignore_recognition_level == 2:
                loss_coeff[path] = hparams.recognition_loss_coeff

            info[path] = info[path][0]

    ignore_description = {
        0: "use all",
        1: "ignore only unmatched_alignment",
        2: "fully ignore recognitio",
    }

    print(" [!] Skip recognition level: {} ({})". \
            format(hparams.ignore_recognition_level,
                   ignore_description[hparams.ignore_recognition_level]))

    for audio_path, text in info.items():
        if hparams.ignore_recognition_level > 0 and loss_coeff[audio_path] != 1:
            continue

        if base_dir not in audio_path:
            audio_path = os.path.join(base_dir, audio_path)

        try:
            tokens = text_to_sequence(text)
        except:
            continue

        fn = partial(
                _process_utterance,
                audio_path, data_dir, tokens, loss_coeff[audio_path])
        futures.append(executor.submit(fn))

    n_frames = [future.result() for future in tqdm(futures)]
    n_frames = [n_frame for n_frame in n_frames if n_frame is not None]

    hours = frames_to_hours(n_frames)

    print(' [*] Loaded metadata for {} examples ({:.2f} hours)'.format(len(n_frames), hours))
    print(' [*] Max length: {}'.format(max(n_frames)))
    print(' [*] Min length: {}'.format(min(n_frames)))

    plot_n_frames(n_frames, os.path.join(
            base_dir, "n_frames_before_filter.png"))

    min_n_frame = hparams.reduction_factor * hparams.min_iters
    max_n_frame = hparams.reduction_factor * hparams.max_iters - hparams.reduction_factor

    n_frames = [n for n in n_frames if min_n_frame <= n <= max_n_frame]
    hours = frames_to_hours(n_frames)

    print(' [*] After filtered: {} examples ({:.2f} hours)'.format(len(n_frames), hours))
    print(' [*] Max length: {}'.format(max(n_frames)))
    print(' [*] Min length: {}'.format(min(n_frames)))

    plot_n_frames(n_frames, os.path.join(
            base_dir, "n_frames_after_filter.png"))

def plot_n_frames(n_frames, path):
    labels, values = list(zip(*Counter(n_frames).most_common()))

    values = [v for _, v in sorted(zip(labels, values))]
    labels = sorted(labels)

    indexes = np.arange(len(labels))
    width = 1

    fig, ax = plt.subplots(figsize=(len(labels) / 2, 5))

    plt.bar(indexes, values, width)
    plt.xticks(indexes + width * 0.5, labels)

    plt.tight_layout()
    plt.savefig(path)


def _process_utterance(audio_path, data_dir, tokens, loss_coeff):
    audio_name = os.path.basename(audio_path)

    filename = audio_name.rsplit('.', 1)[0] + ".npz"
    numpy_path = os.path.join(data_dir, filename)

    if not os.path.exists(numpy_path):
        wav = load_audio(audio_path)

        linear_spectrogram = spectrogram(wav).astype(np.float32)
        mel_spectrogram = melspectrogram(wav).astype(np.float32)

        data = {
            "linear": linear_spectrogram.T,
            "mel": mel_spectrogram.T,
            "tokens": tokens,
            "loss_coeff": loss_coeff,
        }

        n_frame = linear_spectrogram.shape[1]

        if hparams.skip_inadequate:
            min_n_frame = hparams.reduction_factor * hparams.min_iters
            max_n_frame = hparams.reduction_factor * hparams.max_iters - hparams.reduction_factor

            if min_n_frame <= n_frame <= max_n_frame and len(tokens) >= hparams.min_tokens:
                return None

        np.savez(numpy_path, **data, allow_pickle=False)
    else:
        try:
            data = np.load(numpy_path)
            n_frame = data["linear"].shape[0]
        except:
            remove_file(numpy_path)
            return _process_utterance(audio_path, data_dir, tokens, loss_coeff)

    return n_frame

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='spectrogram')

    parser.add_argument('metadata_path', type=str)
    parser.add_argument('--data_dirname', type=str, default="data")
    parser.add_argument('--num_workers', type=int, default=None)

    config = parser.parse_args()
    build_from_path(config)
