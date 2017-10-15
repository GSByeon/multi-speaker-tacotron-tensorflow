import os
import youtube_dl
from pydub import AudioSegment

from utils import makedirs, remove_file


base_dir = os.path.dirname(os.path.realpath(__file__))

def get_mili_sec(text):
    minute, second = text.strip().split(':')
    return (int(minute) * 60 + int(second)) * 1000

class Data(object):
    def __init__(
            self, text_path, video_url, title, start_time, end_time):
        self.text_path = text_path
        self.video_url = video_url
        self.title = title
        self.start = get_mili_sec(start_time)
        self.end = get_mili_sec(end_time)

def read_csv(path):
    with open(path) as f:
        data = []
        for line in f:
            text_path, video_url, title, start_time, end_time = line.split('|')
            data.append(Data(text_path, video_url, title, start_time, end_time))
        return data

def download_audio_with_urls(data, out_ext="wav"):
    for d in data:
        original_path = os.path.join(base_dir, 'audio',
                os.path.basename(d.text_path)).replace('.txt', '.original.mp3')
        out_path = os.path.join(base_dir, 'audio',
                os.path.basename(d.text_path)).replace('.txt', '.wav')

        options = {
            'format': 'bestaudio/best',
            'outtmpl': original_path,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '320',
            }],
        }
        with youtube_dl.YoutubeDL(options) as ydl:
            ydl.download([d.video_url])

        audio = AudioSegment.from_file(original_path)
        audio[d.start:d.end].export(out_path, out_ext)

        remove_file(original_path)

if __name__ == '__main__':
    makedirs(os.path.join(base_dir, "audio"))

    data = read_csv(os.path.join(base_dir, "metadata.csv"))
    download_audio_with_urls(data)
