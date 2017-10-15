import os
import sys
import json
import argparse
import requests
from bs4 import BeautifulSoup
from functools import partial

from utils import download_with_url, makedirs, parallel_run

base_path = os.path.dirname(os.path.realpath(__file__))
RSS_URL = "http://enabler.kbs.co.kr/api/podcast_channel/feed.xml?channel_id=R2010-0440"

def itunes_download(item):
    audio_dir = os.path.join(base_path, "audio")

    date, url = item
    path = os.path.join(audio_dir, "{}.mp4".format(date))

    if not os.path.exists(path):
        download_with_url(url, path)

def download_all(config):
    audio_dir = os.path.join(base_path, "audio")
    makedirs(audio_dir)

    soup = BeautifulSoup(requests.get(RSS_URL).text, "html5lib")

    items = [item for item in soup.find_all('item')]

    titles = [item.find('title').text[9:-3] for item in items]
    guids = [item.find('guid').text for item in items]

    accept_list = ['친절한 인나씨', '반납예정일', '귀욤열매 드세요']

    new_guids = [guid for title, guid in zip(titles, guids) \
            if any(accept in title for accept in accept_list) and '-' not in title]
    new_titles = [title for title, _ in zip(titles, guids) \
            if any(accept in title for accept in accept_list) and '-' not in title]

    for idx, title in enumerate(new_titles):
        print(" [{:3d}] {}, {}".format(idx + 1, title, 
                os.path.basename(new_guids[idx]).split('_')[2]))
        if idx == config.max_num: print("="*30)

    urls = {
            os.path.basename(guid).split('_')[2]: guid \
                    for guid in new_guids[:config.max_num]
    }

    parallel_run(itunes_download, urls.items(),
            desc=" [*] Itunes download", parallel=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_num', default=100, type=int)
    config, unparsed = parser.parse_known_args()

    download_all(config)
