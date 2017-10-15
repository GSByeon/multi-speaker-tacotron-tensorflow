import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.data.python.util import nest
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper \
                import _bahdanau_score, _BaseAttentionMechanism, BahdanauAttention, \
                             AttentionWrapperState, AttentionMechanism

from .modules import prenet

_zero_state_tensors = rnn_cell_impl._zero_state_tensors



class AttentionWrapper(RNNCell):
    """Wraps another `RNNCell` with attention.
    """

    def __init__(self,
                 cell,
                 attention_mechanism,
                 is_manual_attention,
                 manual_alignments,
                 attention_layer_size=None,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
                 initial_cell_state=None,
                 name=None):
        """Construct the `AttentionWrapper`.
        Args:
            cell: An instance of `RNNCell`.
            attention_mechanism: A list of `AttentionMechanism` instances or a single
                instance.
            attention_layer_size: A list of Python integers or a single Python
                integer, the depth of the attention (output) layer(s). If None
                (default), use the context as attention at each time step. Otherwise,
                feed the context and cell output into the attention layer to generate
                attention at each time step. If attention_mechanism is a list,
                attention_layer_size must be a list of the same length.
            alignment_history: Python boolean, whether to store alignment history
                from all time steps in the final output state (currently stored as a
                time major `TensorArray` on which you must call `stack()`).
            cell_input_fn: (optional) A `callable`.    The default is:
                `lambda inputs, attention: array_tf.concat([inputs, attention], -1)`.
            output_attention: Python bool.    If `True` (default), the output at each
                time step is the attention value.    This is the behavior of Luong-style
                attention mechanisms.    If `False`, the output at each time step is
                the output of `cell`.    This is the beahvior of Bhadanau-style
                attention mechanisms.    In both cases, the `attention` tensor is
                propagated to the next time step via the state and is used there.
                This flag only controls whether the attention mechanism is propagated
                up to the next cell in an RNN stack or to the top RNN output.
            initial_cell_state: The initial state value to use for the cell when
                the user calls `zero_state()`.    Note that if this value is provided
                now, and the user uses a `batch_size` argument of `zero_state` which
                does not match the batch size of `initial_cell_state`, proper
                behavior is not guaranteed.
            name: Name to use when creating tf.
        Raises:
            TypeError: `attention_layer_size` is not None and (`attention_mechanism`
                is a list but `attention_layer_size` is not; or vice versa).
            ValueError: if `attention_layer_size` is not None, `attention_mechanism`
                is a list, and its length does not match that of `attention_layer_size`.
        """
        super(AttentionWrapper, self).__init__(name=name)

        self.is_manual_attention = is_manual_attention
        self.manual_alignments = manual_alignments

        if isinstance(attention_mechanism, (list, tuple)):
            self._is_multi = True
            attention_mechanisms = attention_mechanism
            for attention_mechanism in attention_mechanisms:
                if not isinstance(attention_mechanism, AttentionMechanism):
                    raise TypeError(
                            "attention_mechanism must contain only instances of "
                            "AttentionMechanism, saw type: %s"
                            % type(attention_mechanism).__name__)
        else:
            self._is_multi = False
            if not isinstance(attention_mechanism, AttentionMechanism):
                raise TypeError(
                        "attention_mechanism must be an AttentionMechanism or list of "
                        "multiple AttentionMechanism instances, saw type: %s"
                        % type(attention_mechanism).__name__)
            attention_mechanisms = (attention_mechanism,)

        if cell_input_fn is None:
            cell_input_fn = (
                    lambda inputs, attention: tf.concat([inputs, attention], -1))
        else:
            if not callable(cell_input_fn):
                raise TypeError(
                        "cell_input_fn must be callable, saw type: %s"
                        % type(cell_input_fn).__name__)

        if attention_layer_size is not None:
            attention_layer_sizes = tuple(
                    attention_layer_size
                    if isinstance(attention_layer_size, (list, tuple))
                    else (attention_layer_size,))
            if len(attention_layer_sizes) != len(attention_mechanisms):
                raise ValueError(
                        "If provided, attention_layer_size must contain exactly one "
                        "integer per attention_mechanism, saw: %d vs %d"
                        % (len(attention_layer_sizes), len(attention_mechanisms)))
            self._attention_layers = tuple(
                    layers_core.Dense(
                            attention_layer_size, name="attention_layer", use_bias=False)
                    for attention_layer_size in attention_layer_sizes)
            self._attention_layer_size = sum(attention_layer_sizes)
        else:
            self._attention_layers = None
            self._attention_layer_size = sum(
                    attention_mechanism.values.get_shape()[-1].value
                    for attention_mechanism in attention_mechanisms)

        self._cell = cell
        self._attention_mechanisms = attention_mechanisms
        self._cell_input_fn = cell_input_fn
        self._output_attention = output_attention
        self._alignment_history = alignment_history
        with tf.name_scope(name, "AttentionWrapperInit"):
            if initial_cell_state is None:
                self._initial_cell_state = None
            else:
                final_state_tensor = nest.flatten(initial_cell_state)[-1]
                state_batch_size = (
                        final_state_tensor.shape[0].value
                        or tf.shape(final_state_tensor)[0])
                error_message = (
                        "When constructing AttentionWrapper %s: " % self._base_name +
                        "Non-matching batch sizes between the memory "
                        "(encoder output) and initial_cell_state.    Are you using "
                        "the BeamSearchDecoder?    You may need to tile your initial state "
                        "via the tf.contrib.seq2seq.tile_batch function with argument "
                        "multiple=beam_width.")
                with tf.control_dependencies(
                        self._batch_size_checks(state_batch_size, error_message)):
                    self._initial_cell_state = nest.map_structure(
                            lambda s: tf.identity(s, name="check_initial_cell_state"),
                            initial_cell_state)

    def _batch_size_checks(self, batch_size, error_message):
        return [tf.assert_equal(batch_size,
                attention_mechanism.batch_size,
                message=error_message)
                        for attention_mechanism in self._attention_mechanisms]

    def _item_or_tuple(self, seq):
        """Returns `seq` as tuple or the singular element.
        Which is returned is determined by how the AttentionMechanism(s) were passed
        to the constructor.
        Args:
            seq: A non-empty sequence of items or generator.
        Returns:
             Either the values in the sequence as a tuple if AttentionMechanism(s)
             were passed to the constructor as a sequence or the singular element.
        """
        t = tuple(seq)
        if self._is_multi:
            return t
        else:
            return t[0]

    @property
    def output_size(self):
        if self._output_attention:
            return self._attention_layer_size
        else:
            return self._cell.output_size

    @property
    def state_size(self):
        return AttentionWrapperState(
                cell_state=self._cell.state_size,
                time=tf.TensorShape([]),
                attention=self._attention_layer_size,
                alignments=self._item_or_tuple(
                        a.alignments_size for a in self._attention_mechanisms),
                alignment_history=self._item_or_tuple(
                        () for _ in self._attention_mechanisms))    # sometimes a TensorArray

    def zero_state(self, batch_size, dtype):
        with tf.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            if self._initial_cell_state is not None:
                cell_state = self._initial_cell_state
            else:
                cell_state = self._cell.zero_state(batch_size, dtype)
            error_message = (
                    "When calling zero_state of AttentionWrapper %s: " % self._base_name +
                    "Non-matching batch sizes between the memory "
                    "(encoder output) and the requested batch size.    Are you using "
                    "the BeamSearchDecoder?    If so, make sure your encoder output has "
                    "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
                    "the batch_size= argument passed to zero_state is "
                    "batch_size * beam_width.")
            with tf.control_dependencies(
                    self._batch_size_checks(batch_size, error_message)):
                cell_state = nest.map_structure(
                        lambda s: tf.identity(s, name="checked_cell_state"),
                        cell_state)

            return AttentionWrapperState(
                    cell_state=cell_state,
                    time=tf.zeros([], dtype=tf.int32),
                    attention=_zero_state_tensors(self._attention_layer_size, batch_size, dtype),
                    alignments=self._item_or_tuple(
                            attention_mechanism.initial_alignments(batch_size, dtype)
                            for attention_mechanism in self._attention_mechanisms),
                    alignment_history=self._item_or_tuple(
                            tf.TensorArray(dtype=dtype, size=0, dynamic_size=True)
                            if self._alignment_history else ()
                            for _ in self._attention_mechanisms))

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.
        - Step 1: Mix the `inputs` and previous step's `attention` output via
            `cell_input_fn`.
        - Step 2: Call the wrapped `cell` with this input and its previous state.
        - Step 3: Score the cell's output with `attention_mechanism`.
        - Step 4: Calculate the alignments by passing the score through the
            `normalizer`.
        - Step 5: Calculate the context vector as the inner product between the
            alignments and the attention_mechanism's values (memory).
        - Step 6: Calculate the attention output by concatenating the cell output
            and context through the attention layer (a linear layer with
            `attention_layer_size` outputs).
        Args:
            inputs: (Possibly nested tuple of) Tensor, the input at this time step.
            state: An instance of `AttentionWrapperState` containing
                tensors from the previous time step.
        Returns:
            A tuple `(attention_or_cell_output, next_state)`, where:
            - `attention_or_cell_output` depending on `output_attention`.
            - `next_state` is an instance of `AttentionWrapperState`
                 containing the state calculated at this time step.
        Raises:
            TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """
        if not isinstance(state, AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                                            "Received type %s instead."    % type(state))

        # Step 1: Calculate the true inputs to the cell based on the
        # previous attention value.
        cell_inputs = self._cell_input_fn(inputs, state.attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        cell_batch_size = (
                cell_output.shape[0].value or tf.shape(cell_output)[0])
        error_message = (
                "When applying AttentionWrapper %s: " % self.name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the query (decoder output).    Are you using "
                "the BeamSearchDecoder?    You may need to tile your memory input via "
                "the tf.contrib.seq2seq.tile_batch function with argument "
                "multiple=beam_width.")
        with tf.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            cell_output = tf.identity(
                    cell_output, name="checked_cell_output")

        if self._is_multi:
            previous_alignments = state.alignments
            previous_alignment_history = state.alignment_history
        else:
            previous_alignments = [state.alignments]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_histories = []

        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            attention, alignments = _compute_attention(
                    attention_mechanism, cell_output, previous_alignments[i],
                    self._attention_layers[i] if self._attention_layers else None,
                    self.is_manual_attention, self.manual_alignments, state.time)

            alignment_history = previous_alignment_history[i].write(
                    state.time, alignments) if self._alignment_history else ()

            all_alignments.append(alignments)
            all_histories.append(alignment_history)
            all_attentions.append(attention)

        attention = tf.concat(all_attentions, 1)
        next_state = AttentionWrapperState(
                time=state.time + 1,
                cell_state=next_cell_state,
                attention=attention,
                alignments=self._item_or_tuple(all_alignments),
                alignment_history=self._item_or_tuple(all_histories))

        if self._output_attention:
            return attention, next_state
        else:
            return cell_output, next_state

def _compute_attention(
        attention_mechanism, cell_output, previous_alignments,
        attention_layer, is_manual_attention, manual_alignments, time):

    computed_alignments = attention_mechanism(
            cell_output, previous_alignments=previous_alignments)
    batch_size, max_time = \
            tf.shape(computed_alignments)[0], tf.shape(computed_alignments)[1]

    alignments = tf.cond(
            is_manual_attention,
            lambda: manual_alignments[:, time, :],
            lambda: computed_alignments,
    )

    #alignments = tf.one_hot(tf.zeros((batch_size,), dtype=tf.int32), max_time, dtype=tf.float32)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = tf.expand_dims(alignments, 1)

    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #         [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #         [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is
    #         [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.
    context = tf.matmul(expanded_alignments, attention_mechanism.values)
    context = tf.squeeze(context, [1])

    if attention_layer is not None:
        attention = attention_layer(tf.concat([cell_output, context], 1))
    else:
        attention = context

    return attention, alignments


class DecoderPrenetWrapper(RNNCell):
    '''Runs RNN inputs through a prenet before sending them to the cell.'''
    def __init__(
            self, cell, embed_to_concat,
            is_training, prenet_sizes, dropout_prob):

        super(DecoderPrenetWrapper, self).__init__()
        self._is_training = is_training

        self._cell = cell
        self._embed_to_concat = embed_to_concat

        self.prenet_sizes = prenet_sizes
        self.dropout_prob = dropout_prob

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def call(self, inputs, state):
        prenet_out = prenet(
                inputs, self._is_training,
                self.prenet_sizes, self.dropout_prob, scope='decoder_prenet')

        if self._embed_to_concat is not None:
            concat_out = tf.concat(
                    [prenet_out, self._embed_to_concat],
                    axis=-1, name='speaker_concat')
            return self._cell(concat_out, state)
        else:
            return self._cell(prenet_out, state)

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)



class ConcatOutputAndAttentionWrapper(RNNCell):
    '''Concatenates RNN cell output with the attention context vector.

    This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
    attention_layer_size=None and output_attention=False. Such a cell's state will include an
    "attention" field that is the context vector.
    '''
    def __init__(self, cell, embed_to_concat):
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell
        self._embed_to_concat = embed_to_concat

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size + self._cell.state_size.attention

    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)

        if self._embed_to_concat is not None:
            tensors = [
                    output, res_state.attention,
                    self._embed_to_concat,
            ]
            return tf.concat(tensors, axis=-1), res_state
        else:
            return tf.concat([output, res_state.attention], axis=-1), res_state

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)
