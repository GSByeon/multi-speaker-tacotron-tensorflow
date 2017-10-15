# Code based on https://github.com/carpedm20/DCGAN-tensorflow/blob/master/download.py

from __future__ import print_function
import os
import sys
import gzip
import json
import tarfile
import zipfile
import argparse
import requests
from tqdm import tqdm
from six.moves import urllib

from utils import query_yes_no

parser = argparse.ArgumentParser(description='Download model checkpoints.')
parser.add_argument('checkpoints', metavar='N', type=str, nargs='+', choices=['son', 'park'],
                     help='name of checkpoints to download [son, park]')

def download(url, dirpath):
    filename = url.split('/')[-1]
    filepath = os.path.join(dirpath, filename)
    u = urllib.request.urlopen(url)
    f = open(filepath, 'wb')
    filesize = int(u.headers["Content-Length"])
    print("Downloading: %s Bytes: %s" % (filename, filesize))

    downloaded = 0
    block_sz = 8192
    status_width = 70
    while True:
        buf = u.read(block_sz)
        if not buf:
            print('')
            break
        else:
            print('', end='\r')
        downloaded += len(buf)
        f.write(buf)
        status = (("[%-" + str(status_width + 1) + "s] %3.2f%%") %
            ('=' * int(float(downloaded) / filesize * status_width) + '>', downloaded * 100. / filesize))
        print(status, end='')
        sys.stdout.flush()
    f.close()
    return filepath

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={ 'id': id }, stream=True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination, chunk_size=32*1024):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(chunk_size), total=total_size,
                            unit='B', unit_scale=True, desc=destination):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def unzip(filepath):
    print("Extracting: " + filepath)
    dirpath = os.path.dirname(filepath)
    with zipfile.ZipFile(filepath) as zf:
        zf.extractall(dirpath)
    os.remove(filepath)

def download_checkpoint(checkpoint):
    if checkpoint == "son":
        save_path, drive_id = "son-20171015.tar.gz", "0B_7wC-DuR6ORcmpaY1A5V1AzZUU"
    elif checkpoint == "park":
        save_path, drive_id = "park-20171015.tar.gz", "0B_7wC-DuR6ORYjhlekl5bVlkQ2c"
    else:
        raise Exception(" [!] Unknown checkpoint: {}".format(checkpoint))

    if os.path.exists(save_path):
        print('[*] {} already exists'.format(save_path))
    else:
        download_file_from_google_drive(drive_id, save_path)

    if save_path.endswith(".zip"):
        zip_dir = ''
        with zipfile.ZipFile(save_path) as zf:
            zip_dir = zf.namelist()[0]
            zf.extractall(dirpath)
        os.remove(save_path)
        os.rename(os.path.join(dirpath, zip_dir), os.path.join(dirpath, data_dir))
    elif save_path.endswith("tar.gz"):
        tar = tarfile.open(save_path, "r:gz")
        tar.extractall()
        tar.close()
    elif save_path.endswith("tar"):
        tar = tarfile.open(save_path, "r:")
        tar.extractall()
        tar.close()

if __name__ == '__main__':
    args = parser.parse_args()

    print(" [!] The pre-trained models are being made available for research purpose only")
    print(" [!] 학습된 모델을 연구 이외의 목적으로 사용하는 것을 금지합니다.")
    print()

    if query_yes_no(" [?] Are you agree on this? 이에 동의하십니까?"):
        if 'park' in args.checkpoints:
            download_checkpoint('park')
        if 'son' in args.checkpoints:
            download_checkpoint('son')
