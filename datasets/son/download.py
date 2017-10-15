import re
import os
import sys
import m3u8
import json
import requests
import subprocess
from functools import partial
from bs4 import BeautifulSoup

from utils import get_encoder_name, parallel_run, makedirs

API_URL = 'http://api.jtbc.joins.com/ad/pre/NV10173083'
BASE_URL = 'http://nsvc.jtbc.joins.com/API/News/Newapp/Default.aspx'

def soupify(text):
    return BeautifulSoup(text, "html.parser")

def get_news_ids(page_id):
    params = {
        'NJC': 'NJC300',
        'CAID': 'NC10011174',
        'PGI': page_id,
    }

    response = requests.request(
        method='GET', url=BASE_URL, params=params,
    )
    soup = soupify(response.text)

    return [item.text for item in soup.find_all('news_id')]

def download_news_video_and_content(
        news_id, base_dir, chunk_size=32*1024,
        video_dir="video", asset_dir="assets", audio_dir="audio"):

    video_dir = os.path.join(base_dir, video_dir)
    asset_dir = os.path.join(base_dir, asset_dir)
    audio_dir = os.path.join(base_dir, audio_dir)

    makedirs(video_dir)
    makedirs(asset_dir)
    makedirs(audio_dir)

    text_path = os.path.join(asset_dir, "{}.txt".format(news_id))
    original_text_path = os.path.join(asset_dir, "original-{}.txt".format(news_id))

    video_path = os.path.join(video_dir, "{}.ts".format(news_id))
    audio_path = os.path.join(audio_dir, "{}.wav".format(news_id))

    params = {
        'NJC': 'NJC400',
        'NID': news_id, # NB11515152
        'CD': 'A0100',
    }

    response = requests.request(
        method='GET', url=BASE_URL, params=params,
    )
    soup = soupify(response.text)

    article_contents = soup.find_all('article_contents')

    assert len(article_contents) == 1, \
            "# of <article_contents> of {} should be 1: {}".format(news_id, response.text)

    text = soupify(article_contents[0].text).get_text() # remove <div>

    with open(original_text_path, "w") as f:
        f.write(text)

    with open(text_path, "w") as f:
        from nltk import sent_tokenize

        text = re.sub(r'\[.{0,80} :\s.+]', '', text) # remove quote
        text = re.sub(r'☞.+http.+\)', '', text) # remove quote
        text = re.sub(r'\(https?:\/\/.*[\r\n]*\)', '', text) # remove url

        sentences = sent_tokenize(text)
        sentences = [sent for sentence in sentences for sent in sentence.split('\n') if sent]

        new_texts = []
        for sent in sentences:
            sent = sent.strip()
            sent = re.sub(r'\([^)]*\)', '', sent)
            #sent = re.sub(r'\<.{0,80}\>', '', sent)
            sent = sent.replace('…', '.')
            new_texts.append(sent)

        f.write("\n".join([sent for sent in new_texts if sent]))

    vod_paths = soup.find_all('vod_path')

    assert len(vod_paths) == 1, \
            "# of <vod_path> of {} should be 1: {}".format(news_id, response.text)

    if not os.path.exists(video_path):
        redirect_url = soup.find_all('vod_path')[0].text

        list_url = m3u8.load(redirect_url).playlists[0].absolute_uri
        video_urls = [segment.absolute_uri for segment in m3u8.load(list_url).segments]

        with open(video_path, "wb") as f:
            for url in video_urls:
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))

                for chunk in response.iter_content(chunk_size):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)

    if not os.path.exists(audio_path):
        encoder = get_encoder_name()
        command = "{} -y -loglevel panic -i {} -ab 160k -ac 2 -ar 44100 -vn {}".\
                format(encoder, video_path, audio_path)
        subprocess.call(command, shell=True)

    return True

if __name__ == '__main__':
    news_ids = []
    page_idx = 1

    base_dir = os.path.dirname(os.path.realpath(__file__))
    news_id_path = os.path.join(base_dir, "news_ids.json")

    if not os.path.exists(news_id_path):
        while True:
            tmp_ids = get_news_ids(page_idx)
            if len(tmp_ids) == 0:
                break

            news_ids.extend(tmp_ids)
            print(" [*] Download page {}: {}/{}".format(page_idx, len(tmp_ids), len(news_ids)))

            page_idx += 1

        with open(news_id_path, "w") as f:
            json.dump(news_ids, f, indent=2, ensure_ascii=False)
    else:
        with open(news_id_path) as f:
            news_ids = json.loads(f.read())

    exceptions = ["NB10830162"]
    news_ids = list(set(news_ids) - set(exceptions))

    fn = partial(download_news_video_and_content, base_dir=base_dir)

    results = parallel_run(
            fn, news_ids, desc="Download news video+text", parallel=True)
