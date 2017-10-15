import tensorflow as tf

SCALE_FACTOR = 1

def f(num):
    return num // SCALE_FACTOR

basic_params = {
    # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
    # text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
    'cleaners': 'korean_cleaners',
}

basic_params.update({
    # Audio
    'num_mels': 80,
    'num_freq': 1025,
    'sample_rate': 20000,
    'frame_length_ms': 50,
    'frame_shift_ms': 12.5,
    'preemphasis': 0.97,
    'min_level_db': -100,
    'ref_level_db': 20,
})

if True:
    basic_params.update({
        'sample_rate': 24000,
    })

basic_params.update({
    # Model
    'model_type': 'single', # [single, simple, deepvoice]
    'speaker_embedding_size': f(16),

    'embedding_size': f(256),
    'dropout_prob': 0.5,

    # Encoder
    'enc_prenet_sizes': [f(256), f(128)],
    'enc_bank_size': 16,
    'enc_bank_channel_size': f(128),
    'enc_maxpool_width': 2,
    'enc_highway_depth': 4,
    'enc_rnn_size': f(128),
    'enc_proj_sizes': [f(128), f(128)],
    'enc_proj_width': 3,

    # Attention
    'attention_type': 'bah_mon', # ntm2-5
    'attention_size': f(256),
    'attention_state_size': f(256),

    # Decoder recurrent network
    'dec_layer_num': 2,
    'dec_rnn_size': f(256),

    # Decoder
    'dec_prenet_sizes': [f(256), f(128)],
    'post_bank_size': 8,
    'post_bank_channel_size': f(256),
    'post_maxpool_width': 2,
    'post_highway_depth': 4,
    'post_rnn_size': f(128),
    'post_proj_sizes': [f(256), 80], # num_mels=80
    'post_proj_width': 3,

    'reduction_factor': 4,
})

if False: # Deep Voice 2
    basic_params.update({
        'dropout_prob': 0.8,

        'attention_size': f(512),

        'dec_prenet_sizes': [f(256), f(128), f(64)],
        'post_bank_channel_size': f(512),
        'post_rnn_size': f(256),

        'reduction_factor': 4,
    })
elif True: # Deep Voice 2
    basic_params.update({
        'dropout_prob': 0.8,

        #'attention_size': f(512),

        #'dec_prenet_sizes': [f(256), f(128)],
        #'post_bank_channel_size': f(512),
        'post_rnn_size': f(256),

        'reduction_factor': 4,
    })
elif False: # Single Speaker
    basic_params.update({
        'dropout_prob': 0.5,

        'attention_size': f(128),

        'post_bank_channel_size': f(128),
        #'post_rnn_size': f(128),

        'reduction_factor': 4,
    })
elif False: # Single Speaker with generalization
    basic_params.update({
        'dropout_prob': 0.8,

        'attention_size': f(256),

        'dec_prenet_sizes': [f(256), f(128), f(64)],
        'post_bank_channel_size': f(128),
        'post_rnn_size': f(128),

        'reduction_factor': 4,
    })


basic_params.update({
    # Training
    'batch_size': 16,
    'adam_beta1': 0.9,
    'adam_beta2': 0.999,
    'use_fixed_test_inputs': False,

    'initial_learning_rate': 0.002,
    'decay_learning_rate_mode': 0,
    'initial_data_greedy': True,
    'initial_phase_step': 8000,
    'main_data_greedy_factor': 0,
    'main_data': [''],
    'prioritize_loss': False,

    'recognition_loss_coeff': 0.2,
    'ignore_recognition_level': 1, # 0: use all, 1: ignore only unmatched_alignment, 2: fully ignore recognition

    # Eval
    'min_tokens': 50,
    'min_iters': 30,
    'max_iters': 200,
    'skip_inadequate': False,

    'griffin_lim_iters': 60,
    'power': 1.5, # Power to raise magnitudes to prior to Griffin-Lim
})


# Default hyperparameters:
hparams = tf.contrib.training.HParams(**basic_params)


def hparams_debug_string():
    values = hparams.values()
    hp = ['    %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
