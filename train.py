import os
import time
import math
import argparse
import traceback
import subprocess
import numpy as np
from jamo import h2j
import tensorflow as tf
from datetime import datetime
from functools import partial

from hparams import hparams, hparams_debug_string
from models import create_model, get_most_recent_checkpoint

from utils import ValueWindow, prepare_dirs
from utils import infolog, warning, plot, load_hparams
from utils import get_git_revision_hash, get_git_diff, str2bool, parallel_run

from audio import save_audio, inv_spectrogram
from text import sequence_to_text, text_to_sequence
from datasets.datafeeder import DataFeeder, _prepare_inputs

log = infolog.log


def create_batch_inputs_from_texts(texts):
    sequences = [text_to_sequence(text) for text in texts]

    inputs = _prepare_inputs(sequences)
    input_lengths = np.asarray([len(x) for x in inputs], dtype=np.int32)

    for idx, (seq, text) in enumerate(zip(inputs, texts)):
        recovered_text = sequence_to_text(seq, skip_eos_and_pad=True)
        if recovered_text != h2j(text):
            log(" [{}] {}".format(idx, text))
            log(" [{}] {}".format(idx, recovered_text))
            log("="*30)

    return inputs, input_lengths


def get_git_commit():
    subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])     # Verify client is clean
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:10]
    log('Git commit: %s' % commit)
    return commit


def add_stats(model, model2=None, scope_name='train'):
    with tf.variable_scope(scope_name) as scope:
        summaries = [
                tf.summary.scalar('loss_mel', model.mel_loss),
                tf.summary.scalar('loss_linear', model.linear_loss),
                tf.summary.scalar('loss', model.loss_without_coeff),
        ]

        if scope_name == 'train':
            gradient_norms = [tf.norm(grad) for grad in model.gradients if grad is not None]

            summaries.extend([
                    tf.summary.scalar('learning_rate', model.learning_rate),
                    tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms)),
            ])

    if model2 is not None:
        with tf.variable_scope('gap_test-train') as scope:
            summaries.extend([
                    tf.summary.scalar('loss_mel',
                            model.mel_loss - model2.mel_loss),
                    tf.summary.scalar('loss_linear', 
                            model.linear_loss - model2.linear_loss),
                    tf.summary.scalar('loss',
                            model.loss_without_coeff - model2.loss_without_coeff),
            ])

    return tf.summary.merge(summaries)


def save_and_plot_fn(args, log_dir, step, loss, prefix):
    idx, (seq, spec, align) = args

    audio_path = os.path.join(
            log_dir, '{}-step-{:09d}-audio{:03d}.wav'.format(prefix, step, idx))
    align_path = os.path.join(
            log_dir, '{}-step-{:09d}-align{:03d}.png'.format(prefix, step, idx))

    waveform = inv_spectrogram(spec.T)
    save_audio(waveform, audio_path)

    info_text = 'step={:d}, loss={:.5f}'.format(step, loss)
    plot.plot_alignment(
            align, align_path, info=info_text,
            text=sequence_to_text(seq,
                    skip_eos_and_pad=True, combine_jamo=True))

def save_and_plot(sequences, spectrograms,
        alignments, log_dir, step, loss, prefix):

    fn = partial(save_and_plot_fn,
        log_dir=log_dir, step=step, loss=loss, prefix=prefix)
    items = list(enumerate(zip(sequences, spectrograms, alignments)))

    parallel_run(fn, items, parallel=False)
    log('Test finished for step {}.'.format(step))


def train(log_dir, config):
    config.data_paths = config.data_paths

    data_dirs = [os.path.join(data_path, "data") \
            for data_path in config.data_paths]
    num_speakers = len(data_dirs)
    config.num_test = config.num_test_per_speaker * num_speakers

    if num_speakers > 1 and hparams.model_type not in ["deepvoice", "simple"]:
        raise Exception("[!] Unkown model_type for multi-speaker: {}".format(config.model_type))

    commit = get_git_commit() if config.git else 'None'
    checkpoint_path = os.path.join(log_dir, 'model.ckpt')

    log(' [*] git recv-parse HEAD:\n%s' % get_git_revision_hash())
    log('='*50)
    log(' [*] dit diff:\n%s' % get_git_diff())
    log('='*50)
    log(' [*] Checkpoint path: %s' % checkpoint_path)
    log(' [*] Loading training data from: %s' % data_dirs)
    log(' [*] Using model: %s' % config.model_dir)
    log(hparams_debug_string())

    # Set up DataFeeder:
    coord = tf.train.Coordinator()
    with tf.variable_scope('datafeeder') as scope:
        train_feeder = DataFeeder(
                coord, data_dirs, hparams, config, 32,
                data_type='train', batch_size=hparams.batch_size)
        test_feeder = DataFeeder(
                coord, data_dirs, hparams, config, 8,
                data_type='test', batch_size=config.num_test)

    # Set up model:
    is_randomly_initialized = config.initialize_path is None
    global_step = tf.Variable(0, name='global_step', trainable=False)

    with tf.variable_scope('model') as scope:
        model = create_model(hparams)
        model.initialize(
                train_feeder.inputs, train_feeder.input_lengths,
                num_speakers,  train_feeder.speaker_id,
                train_feeder.mel_targets, train_feeder.linear_targets,
                train_feeder.loss_coeff,
                is_randomly_initialized=is_randomly_initialized)

        model.add_loss()
        model.add_optimizer(global_step)
        train_stats = add_stats(model, scope_name='stats') # legacy

    with tf.variable_scope('model', reuse=True) as scope:
        test_model = create_model(hparams)
        test_model.initialize(
                test_feeder.inputs, test_feeder.input_lengths,
                num_speakers, test_feeder.speaker_id,
                test_feeder.mel_targets, test_feeder.linear_targets,
                test_feeder.loss_coeff, rnn_decoder_test_mode=True,
                is_randomly_initialized=is_randomly_initialized)
        test_model.add_loss()

    test_stats = add_stats(test_model, model, scope_name='test')
    test_stats = tf.summary.merge([test_stats, train_stats])

    # Bookkeeping:
    step = 0
    time_window = ValueWindow(100)
    loss_window = ValueWindow(100)
    saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=2)

    sess_config = tf.ConfigProto(
            log_device_placement=False,
            allow_soft_placement=True)
    sess_config.gpu_options.allow_growth=True

    # Train!
    #with tf.Session(config=sess_config) as sess:
    with tf.Session() as sess:
        try:
            summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
            sess.run(tf.global_variables_initializer())

            if config.load_path:
                # Restore from a checkpoint if the user requested it.
                restore_path = get_most_recent_checkpoint(config.model_dir)
                saver.restore(sess, restore_path)
                log('Resuming from checkpoint: %s at commit: %s' % (restore_path, commit), slack=True)
            elif config.initialize_path:
                restore_path = get_most_recent_checkpoint(config.initialize_path)
                saver.restore(sess, restore_path)
                log('Initialized from checkpoint: %s at commit: %s' % (restore_path, commit), slack=True)

                zero_step_assign = tf.assign(global_step, 0)
                sess.run(zero_step_assign)

                start_step = sess.run(global_step)
                log('='*50)
                log(' [*] Global step is reset to {}'. \
                        format(start_step))
                log('='*50)
            else:
                log('Starting new training run at commit: %s' % commit, slack=True)

            start_step = sess.run(global_step)

            train_feeder.start_in_session(sess, start_step)
            test_feeder.start_in_session(sess, start_step)

            while not coord.should_stop():
                start_time = time.time()
                step, loss, opt = sess.run(
                        [global_step, model.loss_without_coeff, model.optimize],
                        feed_dict=model.get_dummy_feed_dict())

                time_window.append(time.time() - start_time)
                loss_window.append(loss)

                message = 'Step %-7d [%.03f sec/step, loss=%.05f, avg_loss=%.05f]' % (
                        step, time_window.average, loss, loss_window.average)
                log(message, slack=(step % config.checkpoint_interval == 0))

                if loss > 100 or math.isnan(loss):
                    log('Loss exploded to %.05f at step %d!' % (loss, step), slack=True)
                    raise Exception('Loss Exploded')

                if step % config.summary_interval == 0:
                    log('Writing summary at step: %d' % step)

                    feed_dict = {
                            **model.get_dummy_feed_dict(),
                            **test_model.get_dummy_feed_dict()
                    }
                    summary_writer.add_summary(sess.run(
                            test_stats, feed_dict=feed_dict), step)

                if step % config.checkpoint_interval == 0:
                    log('Saving checkpoint to: %s-%d' % (checkpoint_path, step))
                    saver.save(sess, checkpoint_path, global_step=step)

                if step % config.test_interval == 0:
                    log('Saving audio and alignment...')
                    num_test = config.num_test

                    fetches = [
                            model.inputs[:num_test],
                            model.linear_outputs[:num_test],
                            model.alignments[:num_test],
                            test_model.inputs[:num_test],
                            test_model.linear_outputs[:num_test],
                            test_model.alignments[:num_test],
                    ]
                    feed_dict = {
                            **model.get_dummy_feed_dict(),
                            **test_model.get_dummy_feed_dict()
                    }

                    sequences, spectrograms, alignments, \
                            test_sequences, test_spectrograms, test_alignments = \
                                    sess.run(fetches, feed_dict=feed_dict)

                    save_and_plot(sequences[:1], spectrograms[:1], alignments[:1],
                            log_dir, step, loss, "train")
                    save_and_plot(test_sequences, test_spectrograms, test_alignments,
                            log_dir, step, loss, "test")

        except Exception as e:
            log('Exiting due to exception: %s' % e, slack=True)
            traceback.print_exc()
            coord.request_stop(e)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--data_paths', default='datasets/kr_example')
    parser.add_argument('--load_path', default=None)
    parser.add_argument('--initialize_path', default=None)

    parser.add_argument('--num_test_per_speaker', type=int, default=2)
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--summary_interval', type=int, default=100)
    parser.add_argument('--test_interval', type=int, default=500)
    parser.add_argument('--checkpoint_interval', type=int, default=1000)
    parser.add_argument('--skip_path_filter',
            type=str2bool, default=False, help='Use only for debugging')

    parser.add_argument('--slack_url',
            help='Slack webhook URL to get periodic reports.')
    parser.add_argument('--git', action='store_true',
            help='If set, verify that the client is clean.')

    config = parser.parse_args()
    config.data_paths = config.data_paths.split(",")
    setattr(hparams, "num_speakers", len(config.data_paths))

    prepare_dirs(config, hparams)

    log_path = os.path.join(config.model_dir, 'train.log')
    infolog.init(log_path, config.model_dir, config.slack_url)

    tf.set_random_seed(config.random_seed)

    if any("krbook" not in data_path for data_path in config.data_paths) and \
            hparams.sample_rate != 20000:
        warning("Detect non-krbook dataset. Set sampling rate from {} to 20000".\
                format(hparams.sample_rate))

    if config.load_path is not None and config.initialize_path is not None:
        raise Exception(" [!] Only one of load_path and initialize_path should be set")

    train(config.model_dir, config)


if __name__ == '__main__':
    main()
