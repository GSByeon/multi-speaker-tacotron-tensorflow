import io
import os
import re
import librosa
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from functools import partial

from hparams import hparams
from models import create_model, get_most_recent_checkpoint
from audio import save_audio, inv_spectrogram, inv_preemphasis, \
                  inv_spectrogram_tensorflow
from utils import plot, PARAMS_NAME, load_json, load_hparams, \
                  add_prefix, add_postfix, get_time, parallel_run, makedirs

from text.korean import tokenize
from text import text_to_sequence, sequence_to_text


class Synthesizer(object):
    def close(self):
        tf.reset_default_graph()
        self.sess.close()

    def load(self, checkpoint_path, num_speakers=2, checkpoint_step=None, model_name='tacotron'):
        self.num_speakers = num_speakers

        if os.path.isdir(checkpoint_path):
            load_path = checkpoint_path
            checkpoint_path = get_most_recent_checkpoint(checkpoint_path, checkpoint_step)
        else:
            load_path = os.path.dirname(checkpoint_path)

        print('Constructing model: %s' % model_name)

        inputs = tf.placeholder(tf.int32, [None, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [None], 'input_lengths')

        batch_size = tf.shape(inputs)[0]
        speaker_id = tf.placeholder_with_default(
                tf.zeros([batch_size], dtype=tf.int32), [None], 'speaker_id')

        load_hparams(hparams, load_path)
        with tf.variable_scope('model') as scope:
            self.model = create_model(hparams)

            self.model.initialize(
                    inputs, input_lengths,
                    self.num_speakers, speaker_id)
            self.wav_output = \
                    inv_spectrogram_tensorflow(self.model.linear_outputs)

        print('Loading checkpoint: %s' % checkpoint_path)

        sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                intra_op_parallelism_threads=1,
                inter_op_parallelism_threads=2)
        sess_config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=sess_config)
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.sess, checkpoint_path)

    def synthesize(self,
            texts=None, tokens=None,
            base_path=None, paths=None, speaker_ids=None,
            start_of_sentence=None, end_of_sentence=True,
            pre_word_num=0, post_word_num=0,
            pre_surplus_idx=0, post_surplus_idx=1,
            use_short_concat=False,
            manual_attention_mode=0,
            base_alignment_path=None,
            librosa_trim=False,
            attention_trim=True):

        # Possible inputs:
        # 1) text=text
        # 2) text=texts
        # 3) tokens=tokens, texts=texts # use texts as guide

        if type(texts) == str:
            texts = [texts]

        if texts is not None and tokens is None:
            sequences = [text_to_sequence(text) for text in texts]
        elif tokens is not None:
            sequences = tokens

        if paths is None:
            paths = [None] * len(sequences)
        if texts is None:
            texts = [None] * len(sequences)

        time_str = get_time()
        def plot_and_save_parallel(
                wavs, alignments, use_manual_attention):

            items = list(enumerate(zip(
                    wavs, alignments, paths, texts, sequences)))

            fn = partial(
                    plot_graph_and_save_audio,
                    base_path=base_path,
                    start_of_sentence=start_of_sentence, end_of_sentence=end_of_sentence,
                    pre_word_num=pre_word_num, post_word_num=post_word_num,
                    pre_surplus_idx=pre_surplus_idx, post_surplus_idx=post_surplus_idx,
                    use_short_concat=use_short_concat,
                    use_manual_attention=use_manual_attention,
                    librosa_trim=librosa_trim,
                    attention_trim=attention_trim,
                    time_str=time_str)
            return parallel_run(fn, items,
                    desc="plot_graph_and_save_audio", parallel=False)

        input_lengths = np.argmax(np.array(sequences) == 1, 1)

        fetches = [
                #self.wav_output,
                self.model.linear_outputs,
                self.model.alignments,
        ]

        feed_dict = {
                self.model.inputs: sequences,
                self.model.input_lengths: input_lengths,
        }
        if base_alignment_path is None:
            feed_dict.update({
                    self.model.manual_alignments: np.zeros([1, 1, 1]),
                    self.model.is_manual_attention: False,
            })
        else:
            manual_alignments = []
            alignment_path = os.path.join(
                    base_alignment_path,
                    os.path.basename(base_path))

            for idx in range(len(sequences)):
                numpy_path = "{}.{}.npy".format(alignment_path, idx)
                manual_alignments.append(np.load(numpy_path))

            alignments_T = np.transpose(manual_alignments, [0, 2, 1])
            feed_dict.update({
                    self.model.manual_alignments: alignments_T,
                    self.model.is_manual_attention: True,
            })

        if speaker_ids is not None:
            if type(speaker_ids) == dict:
                speaker_embed_table = sess.run(
                        self.model.speaker_embed_table)

                speaker_embed =  [speaker_ids[speaker_id] * \
                        speaker_embed_table[speaker_id] for speaker_id in speaker_ids]
                feed_dict.update({
                        self.model.speaker_embed_table: np.tile()
                })
            else:
                feed_dict[self.model.speaker_id] = speaker_ids

        wavs, alignments = \
                self.sess.run(fetches, feed_dict=feed_dict)
        results = plot_and_save_parallel(
                wavs, alignments, True)

        if manual_attention_mode > 0:
            # argmax one hot
            if manual_attention_mode == 1:
                alignments_T = np.transpose(alignments, [0, 2, 1]) # [N, E, D]
                new_alignments = np.zeros_like(alignments_T)

                for idx in range(len(alignments)):
                    argmax = alignments[idx].argmax(1)
                    new_alignments[idx][(argmax, range(len(argmax)))] = 1
            # sharpening
            elif manual_attention_mode == 2:
                new_alignments = np.transpose(alignments, [0, 2, 1]) # [N, E, D]

                for idx in range(len(alignments)):
                    var = np.var(new_alignments[idx], 1)
                    mean_var = var[:input_lengths[idx]].mean()

                    new_alignments = np.pow(new_alignments[idx], 2)
            # prunning
            elif manual_attention_mode == 3:
                new_alignments = np.transpose(alignments, [0, 2, 1]) # [N, E, D]

                for idx in range(len(alignments)):
                    argmax = alignments[idx].argmax(1)
                    new_alignments[idx][(argmax, range(len(argmax)))] = 1

            feed_dict.update({
                    self.model.manual_alignments: new_alignments,
                    self.model.is_manual_attention: True,
            })

            new_wavs, new_alignments = \
                    self.sess.run(fetches, feed_dict=feed_dict)
            results = plot_and_save_parallel(
                    new_wavs, new_alignments, True)

        return results

def plot_graph_and_save_audio(args,
        base_path=None,
        start_of_sentence=None, end_of_sentence=None,
        pre_word_num=0, post_word_num=0,
        pre_surplus_idx=0, post_surplus_idx=1,
        use_short_concat=False,
        use_manual_attention=False, save_alignment=False,
        librosa_trim=False, attention_trim=False,
        time_str=None):

    idx, (wav, alignment, path, text, sequence) = args

    if base_path:
        plot_path = "{}/{}.png".format(base_path, get_time())
    elif path:
        plot_path = path.rsplit('.', 1)[0] + ".png"
    else:
        plot_path = None

    #plot_path = add_prefix(plot_path, time_str)
    if use_manual_attention:
        plot_path = add_postfix(plot_path, "manual")

    if plot_path:
        plot.plot_alignment(alignment, plot_path, text=text)

    if use_short_concat:
        wav = short_concat(
                wav, alignment, text,
                start_of_sentence, end_of_sentence,
                pre_word_num, post_word_num,
                pre_surplus_idx, post_surplus_idx)

    if attention_trim and end_of_sentence:
        end_idx_counter = 0
        attention_argmax = alignment.argmax(0)
        end_idx = min(len(sequence) - 1, max(attention_argmax))
        max_counter = min((attention_argmax == end_idx).sum(), 5)

        for jdx, attend_idx in enumerate(attention_argmax):
            if len(attention_argmax) > jdx + 1:
                if attend_idx == end_idx:
                    end_idx_counter += 1

                if attend_idx == end_idx and attention_argmax[jdx + 1] > end_idx:
                    break

                if end_idx_counter >= max_counter:
                    break
            else:
                break

        spec_end_idx = hparams.reduction_factor * jdx + 3
        wav = wav[:spec_end_idx]

    audio_out = inv_spectrogram(wav.T)

    if librosa_trim and end_of_sentence:
        yt, index = librosa.effects.trim(audio_out,
                frame_length=5120, hop_length=256, top_db=50)
        audio_out = audio_out[:index[-1]]

    if save_alignment:
        alignment_path = "{}/{}.npy".format(base_path, idx)
        np.save(alignment_path, alignment, allow_pickle=False)

    if path or base_path:
        if path:
            current_path = add_postfix(path, idx)
        elif base_path:
            current_path = plot_path.replace(".png", ".wav")

        save_audio(audio_out, current_path)
        return True
    else:
        io_out = io.BytesIO()
        save_audio(audio_out, io_out)
        result = io_out.getvalue()
        return result

def get_most_recent_checkpoint(checkpoint_dir, checkpoint_step=None):
    if checkpoint_step is None:
        checkpoint_paths = [path for path in glob("{}/*.ckpt-*.data-*".format(checkpoint_dir))]
        idxes = [int(os.path.basename(path).split('-')[1].split('.')[0]) for path in checkpoint_paths]

        max_idx = max(idxes)
    else:
        max_idx = checkpoint_step
    lastest_checkpoint = os.path.join(checkpoint_dir, "model.ckpt-{}".format(max_idx))
    print(" [*] Found lastest checkpoint: {}".format(lastest_checkpoint))
    return lastest_checkpoint

def short_concat(
        wav, alignment, text,
        start_of_sentence, end_of_sentence,
        pre_word_num, post_word_num,
        pre_surplus_idx, post_surplus_idx):

    # np.array(list(decomposed_text))[attention_argmax]
    attention_argmax = alignment.argmax(0)

    if not start_of_sentence and pre_word_num > 0:
        surplus_decomposed_text = decompose_ko_text("".join(text.split()[0]))
        start_idx = len(surplus_decomposed_text) + 1

        for idx, attend_idx in enumerate(attention_argmax):
            if attend_idx == start_idx and attention_argmax[idx - 1] < start_idx:
                break

        wav_start_idx = hparams.reduction_factor * idx - 1 - pre_surplus_idx
    else:
        wav_start_idx = 0

    if not end_of_sentence and post_word_num > 0:
        surplus_decomposed_text = decompose_ko_text("".join(text.split()[-1]))
        end_idx = len(decomposed_text.replace(surplus_decomposed_text, '')) - 1

        for idx, attend_idx in enumerate(attention_argmax):
            if attend_idx == end_idx and attention_argmax[idx + 1] > end_idx:
                break

        wav_end_idx = hparams.reduction_factor * idx + 1 + post_surplus_idx
    else:
        if True: # attention based split
            if end_of_sentence:
                end_idx = min(len(decomposed_text) - 1, max(attention_argmax))
            else:
                surplus_decomposed_text = decompose_ko_text("".join(text.split()[-1]))
                end_idx = len(decomposed_text.replace(surplus_decomposed_text, '')) - 1

            while True:
                if end_idx in attention_argmax:
                    break
                end_idx -= 1

            end_idx_counter = 0
            for idx, attend_idx in enumerate(attention_argmax):
                if len(attention_argmax) > idx + 1:
                    if attend_idx == end_idx:
                        end_idx_counter += 1

                    if attend_idx == end_idx and attention_argmax[idx + 1] > end_idx:
                        break

                    if end_idx_counter > 5:
                        break
                else:
                    break

            wav_end_idx = hparams.reduction_factor * idx + 1 + post_surplus_idx
        else:
            wav_end_idx = None

    wav = wav[wav_start_idx:wav_end_idx]

    if end_of_sentence:
        wav = np.lib.pad(wav, ((0, 20), (0, 0)), 'constant', constant_values=0)
    else:
        wav = np.lib.pad(wav, ((0, 10), (0, 0)), 'constant', constant_values=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', required=True)
    parser.add_argument('--sample_path', default="samples")
    parser.add_argument('--text', required=True)
    parser.add_argument('--num_speakers', default=1, type=int)
    parser.add_argument('--speaker_id', default=0, type=int)
    parser.add_argument('--checkpoint_step', default=None, type=int)
    config = parser.parse_args()

    makedirs(config.sample_path)

    synthesizer = Synthesizer()
    synthesizer.load(config.load_path, config.num_speakers, config.checkpoint_step)

    audio = synthesizer.synthesize(
            texts=[config.text],
            base_path=config.sample_path,
            speaker_ids=[config.speaker_id],
            attention_trim=False)[0]
