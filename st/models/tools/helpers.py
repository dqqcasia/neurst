import tensorflow as tf
from tensorflow.contrib.seq2seq import TrainingHelper, CustomHelper, GreedyEmbeddingHelper, InferenceHelper


class ScheduledArgmaxEmbeddingTrainingHelper(TrainingHelper):
    def __init__(self, embedding, sampling_probability, start_tokens, end_token,
                 time_major=False, seed=None, scheduling_seed=None, softmax_temperature=None,
                 name="ScheduledArgmaxEmbeddingTrainingHelper"):
        with tf.name_scope(name, "ScheduledArgmaxEmbeddingTrainingHelper",
                           [embedding, sampling_probability]):
            if callable(embedding):
                self._embedding_fn = embedding
            else:
                self._embedding_fn = (lambda ids: tf.nn.embedding_lookup(embedding, ids))
            self._start_tokens = tf.convert_to_tensor(
                start_tokens, dtype=tf.int32, name="start_tokens")
            self._end_token = tf.convert_to_tensor(
                end_token, dtype=tf.int32, name="end_token")
            if self._start_tokens.get_shape().ndims != 1:
                raise ValueError("start_tokens must be a vector")
            self._batch_size = tf.size(start_tokens)
            if self._end_token.get_shape().ndims != 0:
                raise ValueError("end_token must be a scalar")
            self._start_inputs = self._embedding_fn(self._start_tokens)

            self._sampling_probability = tf.convert_to_tensor(sampling_probability,
                                                              name="sampling_probability")
            if self._sampling_probability.get_shape().ndims not in (0, 1):
                raise ValueError(
                    "sampling_probability must be either a scalar or a vector. "
                    "saw shape: %s" % (self._sampling_probability.get_shape()))
            self._softmax_temperature = softmax_temperature
            self._seed = seed
            self._scheduling_seed = scheduling_seed

    def initialize(self, name=None):
        with tf.name_scope(name, "TrainingHelperInitialize"):
            finished = tf.tile([False], [self._batch_size])

            return (finished, self._start_inputs)

    def sample(self, time, outputs, state, name=None):
        with tf.name_scope(name, "ScheduledArgmaxEmbeddingTrainingHelperSample",
                            [time, outputs, state]):
            if self._softmax_temperature is None:
                logits = outputs
            else:
                logits = outputs / self._softmax_temperature
            sample_ids = tf.distributions.Categorical(logits=logits).sample(seed=self._seed)

            policy = tf.distributions.Bernoulli(probs=self._sampling_probability, dtype=tf.bool).\
                sample(sample_shape=self.batch_size, seed=self._scheduling_seed)
            selected = tf.where(policy, sample_ids, sample_ids)

            return selected

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with tf.name_scope(name, "ScheduledArgmaxEmbeddingTrainingHelperNextInputs",
                           [time, outputs, state, sample_ids]):
            del time, outputs
            finished = tf.equal(sample_ids, self._end_token)
            all_finished = tf.reduce_all(finished)
            next_inputs = tf.cond(
                all_finished,
                lambda: self._start_inputs,
                lambda: self._embedding_fn(sample_ids))

            return (finished, next_inputs, state)


class RNAGreedyEmbeddingHelper(GreedyEmbeddingHelper):
    """
    don't need the labels(and you can't give the label as you don't know the alignment)
    so we can use the helper both in the training and infer phrases.
    """
    def __init__(self, encoded, len_encoded, embedding, start_tokens):
        if callable(embedding):
            self._embedding_fn = embedding
        else:
            self._embedding_fn = (
                lambda ids: tf.nn.embedding_lookup(embedding, ids))

        self._start_tokens = tf.convert_to_tensor(
            start_tokens, dtype=tf.int32, name="start_tokens")
        if self._start_tokens.get_shape().ndims != 1:
            raise ValueError("start_tokens must be a vector")
        self._batch_size = tf.size(start_tokens)
        self._encoded = encoded
        self._sequence_length = len_encoded
        self._start_inputs = tf.concat(
            [encoded[:, 0, :], self._embedding_fn(self._start_tokens)], -1)

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        with tf.name_scope(name, "RNANextInputs", [time, outputs, state]):
            next_time = time + 1
            finished = (next_time >= self._sequence_length)
            all_finished = tf.reduce_all(finished)
            inputs = tf.concat([self._encoded[:, time, :], self._embedding_fn(sample_ids)], -1)
            next_inputs = tf.cond(
                all_finished,
                # If we're finished, the next_inputs value doesn't matter
                lambda: self._start_inputs,
                lambda: inputs)
        return (finished, next_inputs, state)


class RNASampleEmbeddingHelper(RNAGreedyEmbeddingHelper):
    """
    don't need the labels(and you can't give the label as you don't know the alignment)
    so we can use the helper both in the training and infer phrases.
    """

    def __init__(self, encoded, len_encoded, embedding, start_tokens, softmax_temperature):
        self._softmax_temperature = softmax_temperature
        self._seed = None
        super().__init__(encoded, len_encoded, embedding, start_tokens)

    def sample(self, time, outputs, state, name=None):
        """sample for GreedyEmbeddingHelper."""
        del time, state  # unused by sample_fn
        if not isinstance(outputs, tf.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))

        if self._softmax_temperature is None:
            logits = outputs
        else:
            logits = outputs / self._softmax_temperature
        sample_ids = tf.distributions.Categorical(logits=logits).sample(seed=self._seed)

        return sample_ids