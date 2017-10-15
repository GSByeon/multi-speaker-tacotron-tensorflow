import os
import datetime
from glob import glob
from tqdm import tqdm
from tinytag import TinyTag
from collections import defaultdict
from multiprocessing.dummy import Pool

from utils import load_json

def second_to_hour(sec):
    return str(datetime.timedelta(seconds=int(sec)))

def get_duration(path):
    filename = os.path.basename(path)
    candidates = filename.split('.')[0].split('_')
    dataset = candidates[0]

    if not os.path.exists(path):
        print(" [!] {} not found".format(path))
        return dataset, 0

    if True: # tinytag
        tag = TinyTag.get(path)
        duration = tag.duration
    else: # librosa
        y, sr = librosa.load(path)
        duration = librosa.get_duration(y=y, sr=sr)

    return dataset, duration

def get_durations(paths, print_detail=True):
    duration_all = 0
    duration_book = defaultdict(list)

    pool = Pool()
    iterator = pool.imap_unordered(get_duration, paths)
    for dataset, duration in tqdm(iterator, total=len(paths)):
        duration_all += duration
        duration_book[dataset].append(duration)

    total_count = 0
    for book, duration in duration_book.items():
        if book:
            time = second_to_hour(sum(duration))
            file_count = len(duration)
            total_count += file_count

            if print_detail:
                print(" [*] Duration of {}: {} (file #: {})". \
                        format(book, time, file_count))

    print(" [*] Total Duration : {} (file #: {})". \
            format(second_to_hour(duration_all), total_count))
    print()
    return duration_all


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-pattern', default=None) # datasets/krbook/audio/*.wav
    parser.add_argument('--data-path', default=None) # datasets/jtbc/alignment.json
    config, unparsed = parser.parse_known_args()

    if config.audio_pattern is not None:
        duration = get_durations(get_paths_by_pattern(config.data_dir))
    elif config.data_path is not None:
        paths = load_json(config.data_path).keys()
        duration = get_durations(paths)
