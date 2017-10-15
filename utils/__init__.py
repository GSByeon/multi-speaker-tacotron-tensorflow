import os
import re
import sys
import json
import requests
import subprocess
from tqdm import tqdm
from contextlib import closing
from multiprocessing import Pool
from collections import namedtuple
from datetime import datetime, timedelta
from shutil import copyfile as copy_file

PARAMS_NAME = "params.json"

class ValueWindow():
    def __init__(self, window_size=100):
        self._window_size = window_size
        self._values = []

    def append(self, x):
        self._values = self._values[-(self._window_size - 1):] + [x]

    @property
    def sum(self):
        return sum(self._values)

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        return self.sum / max(1, self.count)

    def reset(self):
        self._values = []

def prepare_dirs(config, hparams):
    if hasattr(config, "data_paths"):
        config.datasets = [
                os.path.basename(data_path) for data_path in config.data_paths]
        dataset_desc = "+".join(config.datasets)

    if config.load_path:
        config.model_dir = config.load_path
    else:
        config.model_name = "{}_{}".format(dataset_desc, get_time())
        config.model_dir = os.path.join(config.log_dir, config.model_name)

        for path in [config.log_dir, config.model_dir]:
            if not os.path.exists(path):
                os.makedirs(path)

    if config.load_path:
        load_hparams(hparams, config.model_dir)
    else:
        setattr(hparams, "num_speakers", len(config.datasets))

        save_hparams(config.model_dir, hparams)
        copy_file("hparams.py", os.path.join(config.model_dir, "hparams.py"))

def makedirs(path):
    if not os.path.exists(path):
        print(" [*] Make directories : {}".format(path))
        os.makedirs(path)

def remove_file(path):
    if os.path.exists(path):
        print(" [*] Removed: {}".format(path))
        os.remove(path)

def backup_file(path):
    root, ext = os.path.splitext(path)
    new_path = "{}.backup_{}{}".format(root, get_time(), ext)

    os.rename(path, new_path)
    print(" [*] {} has backup: {}".format(path, new_path))

def get_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True, ensure_ascii=False)

def load_json(path, as_class=False):
    with open(path) as f:
        content = f.read()
        content = re.sub(",\s*}", "}", content)
        content = re.sub(",\s*]", "]", content)

        if as_class:
            data = json.loads(content, object_hook=\
                    lambda data: namedtuple('Data', data.keys())(*data.values()))
        else:
            data = json.loads(content)
    return data

def save_hparams(model_dir, hparams):
    param_path = os.path.join(model_dir, PARAMS_NAME)

    info = eval(hparams.to_json(). \
            replace('true', 'True').replace('false', 'False'))
    write_json(param_path, info)

    print(" [*] MODEL dir: {}".format(model_dir))
    print(" [*] PARAM path: {}".format(param_path))

def load_hparams(hparams, load_path, skip_list=[]):
    path = os.path.join(load_path, PARAMS_NAME)

    new_hparams = load_json(path)
    hparams_keys = vars(hparams).keys()

    for key, value in new_hparams.items():
        if key in skip_list or key not in hparams_keys:
            print("Skip {} because it not exists".format(key))
            continue

        if key not in ['job_name', 'num_workers', 'display', 'is_train', 'load_path'] or \
                key == "pointer_load_path":
            original_value = getattr(hparams, key)
            if original_value != value:
                print("UPDATE {}: {} -> {}".format(key, getattr(hparams, key), value))
                setattr(hparams, key, value)

def add_prefix(path, prefix):
    dir_path, filename = os.path.dirname(path), os.path.basename(path)
    return "{}/{}.{}".format(dir_path, prefix, filename)

def add_postfix(path, postfix):
    path_without_ext, ext = path.rsplit('.', 1)
    return "{}.{}.{}".format(path_without_ext, postfix, ext)

def remove_postfix(path):
    items = path.rsplit('.', 2)
    return items[0] + "." + items[2]

def parallel_run(fn, items, desc="", parallel=True):
    results = []

    if parallel:
        with closing(Pool()) as pool:
            for out in tqdm(pool.imap_unordered(
                    fn, items), total=len(items), desc=desc):
                if out is not None:
                    results.append(out)
    else:
        for item in tqdm(items, total=len(items), desc=desc):
            out = fn(item)
            if out is not None:
                results.append(out)

    return results

def which(program):
    if os.name == "nt" and not program.endswith(".exe"):
        program += ".exe"

    envdir_list = [os.curdir] + os.environ["PATH"].split(os.pathsep)

    for envdir in envdir_list:
        program_path = os.path.join(envdir, program)
        if os.path.isfile(program_path) and os.access(program_path, os.X_OK):
            return program_path

def get_encoder_name():
    if which("avconv"):
        return "avconv"
    elif which("ffmpeg"):
        return "ffmpeg"
    else:
        return "ffmpeg"

def download_with_url(url, dest_path, chunk_size=32*1024):
    with open(dest_path, "wb") as f:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        for chunk in response.iter_content(chunk_size):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    return True

def str2bool(v):
    return v.lower() in ('true', '1')

def get_git_revision_hash():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8")

def get_git_diff():
    return subprocess.check_output(['git', 'diff']).decode("utf-8")

def warning(msg):
    print("="*40)
    print(" [!] {}".format(msg))
    print("="*40)
    print()

def query_yes_no(question, default=None):
    # Code from https://stackoverflow.com/a/3041990
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")
