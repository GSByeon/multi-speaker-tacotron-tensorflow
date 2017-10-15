#!flask/bin/python
import os
import hashlib
import argparse
from flask_cors import CORS
from flask import Flask, request, render_template, jsonify, \
        send_from_directory, make_response, send_file

from hparams import hparams
from audio import load_audio
from synthesizer import Synthesizer
from utils import str2bool, prepare_dirs, makedirs, add_postfix

ROOT_PATH = "web"
AUDIO_DIR = "audio"
AUDIO_PATH = os.path.join(ROOT_PATH, AUDIO_DIR)

base_path = os.path.dirname(os.path.realpath(__file__))
static_path = os.path.join(base_path, 'web/static')

global_config = None
synthesizer = Synthesizer()
app = Flask(__name__, root_path=ROOT_PATH, static_url_path='')
CORS(app)


def match_target_amplitude(sound, target_dBFS):
   change_in_dBFS = target_dBFS - sound.dBFS
   return sound.apply_gain(change_in_dBFS)

def amplify(path, keep_silence=300):
    sound = AudioSegment.from_file(path)

    nonsilent_ranges = pydub.silence.detect_nonsilent(
            sound, silence_thresh=-50, min_silence_len=300)

    new_sound = None
    for idx, (start_i, end_i) in enumerate(nonsilent_ranges):
        if idx == len(nonsilent_ranges) - 1:
            end_i = None

        amplified_sound = \
                match_target_amplitude(sound[start_i:end_i], -20.0)

        if idx == 0:
            new_sound = amplified_sound
        else:
            new_sound = new_sound.append(amplified_sound)

        if idx < len(nonsilent_ranges) - 1:
            new_sound = new_sound.append(sound[end_i:nonsilent_ranges[idx+1][0]])

    return new_sound.export("out.mp3", format="mp3")

def generate_audio_response(text, speaker_id):
    global global_config

    model_name = os.path.basename(global_config.load_path)
    hashed_text = hashlib.md5(text.encode('utf-8')).hexdigest()

    relative_dir_path = os.path.join(AUDIO_DIR, model_name)
    relative_audio_path = os.path.join(
            relative_dir_path, "{}.{}.wav".format(hashed_text, speaker_id))
    real_path = os.path.join(ROOT_PATH, relative_audio_path)
    makedirs(os.path.dirname(real_path))

    if not os.path.exists(add_postfix(real_path, 0)):
        try:
            audio = synthesizer.synthesize(
                    [text], paths=[real_path], speaker_ids=[speaker_id],
                    attention_trim=True)[0]
        except:
            return jsonify(success=False), 400

    return send_file(
            add_postfix(relative_audio_path, 0),
            mimetype="audio/wav", 
            as_attachment=True, 
            attachment_filename=hashed_text + ".wav")

    response = make_response(audio)
    response.headers['Content-Type'] = 'audio/wav'
    response.headers['Content-Disposition'] = 'attachment; filename=sound.wav'
    return response

@app.route('/')
def index():
    text = request.args.get('text') or "듣고 싶은 문장을 입력해 주세요."
    return render_template('index.html', text=text)

@app.route('/generate')
def view_method():
    text = request.args.get('text')
    speaker_id = int(request.args.get('speaker_id'))

    if text:
        return generate_audio_response(text, speaker_id)
    else:
        return {}

@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory(
            os.path.join(static_path, 'js'), path)

@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory(
            os.path.join(static_path, 'css'), path)

@app.route('/audio/<path:path>')
def send_audio(path):
    return send_from_directory(
            os.path.join(static_path, 'audio'), path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', required=True)
    parser.add_argument('--checkpoint_step', default=None, type=int)
    parser.add_argument('--num_speakers', default=1, type=int)
    parser.add_argument('--port', default=5000, type=int)
    parser.add_argument('--debug', default=False, type=str2bool)
    config = parser.parse_args()

    if os.path.exists(config.load_path):
        prepare_dirs(config, hparams)

        global_config = config
        synthesizer.load(config.load_path, config.num_speakers, config.checkpoint_step)
    else:
        print(" [!] load_path not found: {}".format(config.load_path))

    app.run(host='0.0.0.0', port=config.port, debug=config.debug)
