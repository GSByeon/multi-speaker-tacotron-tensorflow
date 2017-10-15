# Code based on https://github.com/keithito/tacotron/blob/master/models/tacotron.py

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.layers import core
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper \
        import _bahdanau_score, _BaseAttentionMechanism, BahdanauAttention, \
               AttentionWrapper, AttentionWrapperState


def get_embed(inputs, num_inputs, embed_size, name):
    embed_table = tf.get_variable(
            name, [num_inputs, embed_size], dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(stddev=0.1))
    return tf.nn.embedding_lookup(embed_table, inputs)


def prenet(inputs, is_training, layer_sizes, drop_prob, scope=None):
    x = inputs
    drop_rate = drop_prob if is_training else 0.0
    with tf.variable_scope(scope or 'prenet'):
        for i, size in enumerate(layer_sizes):
            dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i+1))
            x = tf.layers.dropout(dense, rate=drop_rate, name='dropout_%d' % (i+1))
    return x

def cbhg(inputs, input_lengths, is_training, 
        bank_size, bank_channel_size,
        maxpool_width, highway_depth, rnn_size,
        proj_sizes, proj_width, scope,
        before_highway=None, encoder_rnn_init_state=None):

    batch_size = tf.shape(inputs)[0]
    with tf.variable_scope(scope):
        with tf.variable_scope('conv_bank'):
            # Convolution bank: concatenate on the last axis
            # to stack channels from all convolutions
            conv_fn = lambda k: \
                    conv1d(inputs, k, bank_channel_size, 
                            tf.nn.relu, is_training, 'conv1d_%d' % k)

            conv_outputs = tf.concat(
                [conv_fn(k) for k in range(1, bank_size+1)], axis=-1,
            )

        # Maxpooling:
        maxpool_output = tf.layers.max_pooling1d(
            conv_outputs,
            pool_size=maxpool_width,
            strides=1,
            padding='same')

        # Two projection layers:
        proj_out = maxpool_output
        for idx, proj_size in enumerate(proj_sizes):
            activation_fn = None if idx == len(proj_sizes) - 1 else tf.nn.relu
            proj_out = conv1d(
                    proj_out, proj_width, proj_size, activation_fn,
                    is_training, 'proj_{}'.format(idx + 1))

        # Residual connection:
        if before_highway is not None:
            expanded_before_highway = tf.expand_dims(before_highway, [1])
            tiled_before_highway = tf.tile(
                    expanded_before_highway, [1, tf.shape(proj_out)[1], 1])

            highway_input = proj_out + inputs + tiled_before_highway
        else:
            highway_input = proj_out + inputs

        # Handle dimensionality mismatch:
        if highway_input.shape[2] != rnn_size:
            highway_input = tf.layers.dense(highway_input, rnn_size)

        # 4-layer HighwayNet:
        for idx in range(highway_depth):
            highway_input = highwaynet(highway_input, 'highway_%d' % (idx+1))

        rnn_input = highway_input

        # Bidirectional RNN
        if encoder_rnn_init_state is not None:
            initial_state_fw, initial_state_bw = \
                    tf.split(encoder_rnn_init_state, 2, 1)
        else:
            initial_state_fw, initial_state_bw = None, None

        cell_fw, cell_bw = GRUCell(rnn_size), GRUCell(rnn_size)
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw,
                rnn_input,
                sequence_length=input_lengths,
                initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw,
                dtype=tf.float32)
        return tf.concat(outputs, axis=2)    # Concat forward and backward


def batch_tile(tensor, batch_size):
    expaneded_tensor = tf.expand_dims(tensor, [0])
    return tf.tile(expaneded_tensor, \
            [batch_size] + [1 for _ in tensor.get_shape()])


def highwaynet(inputs, scope):
    highway_dim = int(inputs.get_shape()[-1])

    with tf.variable_scope(scope):
        H = tf.layers.dense(
            inputs,
            units=highway_dim,
            activation=tf.nn.relu,
            name='H')
        T = tf.layers.dense(
            inputs,
            units=highway_dim,
            activation=tf.nn.sigmoid,
            name='T',
            bias_initializer=tf.constant_initializer(-1.0))
        return H * T + inputs * (1.0 - T)


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
    with tf.variable_scope(scope):
        conv1d_output = tf.layers.conv1d(
            inputs,
            filters=channels,
            kernel_size=kernel_size,
            activation=activation,
            padding='same')
        return tf.layers.batch_normalization(conv1d_output, training=is_training)
