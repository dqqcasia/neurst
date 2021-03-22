import tensorflow as tf
import logging
import sys
from collections import namedtuple
from tensorflow.contrib.layers import fully_connected

from st.tools.tftools.gradientTools import average_gradients, handle_gradients
from st.models.model_tools import warmup_exponential_decay, choose_device, lr_decay_with_warmup, stepped_down_decay, exponential_decay
from st.models.layers import build_cell, cell_forward
from st.layers.common_layers_v1 import layer_norm


class LSTM_Model(object):
    num_Instances = 0
    num_Model = 0

    def __init__(self, tensor_global_step, is_train, args, batch=None, name='model'):
        # Initialize some parameters
        self.is_train = is_train
        self.num_gpus = args.num_gpus if is_train else 1
        self.list_gpu_devices = args.list_gpus
        self.center_device = "/cpu:0"
        self.learning_rate = None
        self.args = args
        self.batch = batch
        self.name = name

        if batch:
            if args.model.use_multilabel:
                self.build_input = self.build_tf_input_multilabel
            elif args.model.use_bert:
                self.build_input = self.build_tf_input_kdbert
            elif args.model.use_ctc_bert:
                self.build_input = self.build_tf_input_ctcbert
            elif args.model.use_ctc_bertfull:
                self.build_input = self.build_tf_input_ctcbertfull
            elif args.model.is_multilingual:
                self.build_input = self.build_tf_input_multilingual
            elif args.model.is_cotrain:
                self.build_input = self.build_tf_input_multilingual_multilabel
            else:
                self.build_input = self.build_tf_input
        else:

            if args.model.is_multilingual or args.model.is_cotrain:

                self.build_input = self.build_pl_input_multilingual

            elif args.model.use_multilabel:
                self.build_input = self.build_pl_input_multilabel
            else:
                self.build_input = self.build_pl_input


        self.list_pl = None

        self.global_step = tensor_global_step

        # Build graph
        self.list_run = list(self.build_graph() if is_train else self.build_infer_graph())

    def build_graph(self):
        # cerate input tensors in the cpu
        tensors_input = self.build_input()
        # create optimizer
        self.optimizer = self.build_optimizer()
        # if 'horovod' in sys.modules:
        if self.args.use_horovod:
            import horovod.tensorflow as hvd
            logging.info('wrap the optimizer with horovod!')
            self.optimizer = hvd.DistributedOptimizer(self.optimizer)

        loss_step = []
        tower_grads = []
        list_debug = []

        cache = {}
        # the outer scope is necessary for the where the reuse scope need to be limited whthin
        # or reuse=tf.get_variable_scope().reuse
        for id_gpu, name_gpu in enumerate(self.list_gpu_devices):

            def daisy_chain_getter(getter, name, *args, **kwargs):

                """Get a variable and cache in a daisy chain."""
                device_var_key = (name_gpu, name)
                if device_var_key in cache:
                    # if we have the variable on the correct device, return it.
                    return cache[device_var_key]
                if name in cache:
                    # if we have it on a different device, copy it from the last device
                    v = tf.identity(cache[name])
                else:
                    var = getter(name, *args, **kwargs)
                    v = tf.identity(var._ref())  # pylint: disable=protected-access
                # update the cache
                cache[name] = v
                cache[device_var_key] = v
                return v
            # self._use_daisy_chain_getter = True

            # custom_getter = daisy_chain_getter if self._use_daisy_chain_getter else None

            # with tf.variable_scope(self.name, custom_getter=custom_getter, reuse=tf.AUTO_REUSE):
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                loss, gradients, debug = self.build_single_graph(id_gpu, name_gpu, tensors_input)
                loss_step.append(loss)
                tower_grads.append(gradients)
                list_debug.append(debug)

        # mean the loss
        loss = tf.reduce_mean(loss_step)
        # merge gradients, update current model
        with tf.device(self.center_device):
            # computation relevant to gradient
            averaged_grads = average_gradients(tower_grads)
            handled_grads = handle_gradients(averaged_grads, self.args)

            # filter frozen variables:
            if self.args.frozen_scope:
                filtered_grads = []
                frozen_layers = self.args.frozen_scope.split(',')
                for grad, var in handled_grads:
                    flag = False
                    for layer in frozen_layers:
                        if layer in var.name:
                            flag = True
                    if not flag:
                        filtered_grads.append([grad, var])
                        print([grad, var])
            else:
                filtered_grads = handled_grads
            op_optimize = self.optimizer.apply_gradients(filtered_grads, self.global_step)

        self.__class__.num_Instances += 1
        logging.info("built {} {} instance(s).".format(self.__class__.num_Instances, self.__class__.__name__))
        self.summary_op = tf.summary.merge_all()
        self.loss = loss
        return loss, tensors_input.shape_batch, op_optimize, [x for x in zip(*list_debug)]

    def build_infer_graph(self):
        """
        reuse=True if build train models above
        reuse=False if in the inder file
        """
        # cerate input tensors in the cpu
        tensors_input = self.build_input()

        # the outer scope is necessary for the where the reuse scope need to be limited whthin
        # or reuse=tf.get_variable_scope().reuse
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            loss, logits, _ = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)

        # TODO havn't checked
        infer = tf.nn.in_top_k(logits, tf.reshape(tensors_input.label_splits[0], [-1]), 1)
        return loss, tensors_input.shape_batch, infer

    def build_pl_input(self):
        """
        use for training. but recomend to use build_tf_input insted
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, label_splits, len_fea_splits, len_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                batch_features = tf.placeholder(tf.float32, [None, None, self.args.data.dim_input], name='input_feature')
                batch_labels = tf.placeholder(tf.int32, [None, None], name='input_labels')
                batch_fea_lens = tf.placeholder(tf.int32, [None], name='input_fea_lens')
                batch_label_lens = tf.placeholder(tf.int32, [None], name='input_label_lens')
                self.list_pl = [batch_features, batch_labels, batch_fea_lens, batch_label_lens]
                # split input data alone batch axis to gpus

                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(batch_labels, self.num_gpus, name="label_splits")
                tensors_input.len_fea_splits = tf.split(batch_fea_lens, self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(batch_label_lens, self.num_gpus, name="len_label_splits")

        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input

    def build_pl_input_multilingual(self):
        """
                use for training. but recomend to use build_tf_input insted
                """
        tensors_input = namedtuple('tensors_input',
                                   'feature_splits, label_splits, len_fea_splits, len_label_splits, lang_label_splits, shape_batch')


        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                batch_features = tf.placeholder(tf.float32, [None, None, self.args.data.dim_input], name='input_feature')
                batch_labels = tf.placeholder(tf.int32, [None, None], name='input_labels')
                batch_fea_lens = tf.placeholder(tf.int32, [None], name='input_fea_lens')
                batch_label_lens = tf.placeholder(tf.int32, [None], name='input_label_lens')
                batch_lang_labels = tf.placeholder(tf.int32, [None, None], name='input_lang_labels')
                self.list_pl = [batch_features, batch_labels, batch_fea_lens, batch_label_lens, batch_lang_labels]
                # split input data alone batch axis to gpus
                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(batch_labels, self.num_gpus, name="label_splits")
                tensors_input.len_fea_splits = tf.split(batch_fea_lens, self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(batch_label_lens, self.num_gpus, name="len_label_splits")
                tensors_input.lang_label_splits = tf.split(batch_lang_labels, self.num_gpus, name="lang_label_splits")

        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input

    def build_pl_input_multilabel(self):
        """
        use for training. but recomend to use build_tf_input insted
        """
        tensors_input = namedtuple('tensors_input',
                                   'feature_splits, label_splits, aux_label_splits, len_fea_splits, len_label_splits, len_aux_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                batch_features = tf.placeholder(tf.float32, [None, None, self.args.data.dim_input], name='input_feature')
                batch_labels = tf.placeholder(tf.int32, [None, None], name='input_labels')
                batch_aux_labels = tf.placeholder(tf.int32, [None, None], name='batch_aux_labels')

                batch_fea_lens = tf.placeholder(tf.int32, [None], name='input_fea_lens')
                batch_label_lens = tf.placeholder(tf.int32, [None], name='input_label_lens')
                batch_aux_label_lens = tf.placeholder(tf.int32, [None], name='batch_aux_lens')

                self.list_pl = [batch_features, batch_labels, batch_aux_labels, batch_fea_lens, batch_label_lens,
                                batch_aux_label_lens]

                # split input data alone batch axis to gpus
                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(batch_labels, self.num_gpus, name="label_splits")
                tensors_input.aux_label_splits = tf.split(batch_aux_labels, self.num_gpus, name="aux_label_splits")

                tensors_input.len_fea_splits = tf.split(batch_fea_lens, self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(batch_label_lens, self.num_gpus, name="len_label_splits")
                tensors_input.len_aux_label_splits = tf.split(batch_aux_label_lens, self.num_gpus, name='len_aux_label_splits')

        tensors_input.shape_batch = tf.shape(batch_features)


        return tensors_input

    def build_infer_input(self):
        """
        used for inference. For inference must use placeholder.
        during the infer, we only get the decoded result and not use label
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, len_fea_splits, label_splits, len_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                batch_features = tf.placeholder(tf.float32, [None, None, self.args.data.dim_input], name='input_feature')
                batch_fea_lens = tf.placeholder(tf.int32, [None], name='input_fea_lens')
                self.list_pl = [batch_features, batch_fea_lens]
                # split input data alone batch axis to gpus
                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.len_fea_splits = tf.split(batch_fea_lens, self.num_gpus, name="len_fea_splits")

        tensors_input.label_splits = None
        tensors_input.len_label_splits = None
        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input

    def build_tf_input(self):
        """
        stand training input
        """
        tensors_input = namedtuple('tensors_input',
            'id_splits, feature_splits, label_splits, len_fea_splits, len_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                # split input data alone batch axis to gpus
                tensors_input.id_splits = tf.split(self.batch[0], self.num_gpus, name="id_splits")
                tensors_input.feature_splits = tf.split(self.batch[1], self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(self.batch[2], self.num_gpus, name="label_splits")
                tensors_input.len_fea_splits = tf.split(self.batch[3], self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(self.batch[4], self.num_gpus, name="len_label_splits")
                if self.args.model.is_multilingual:
                    tensors_input.lang_id_splits = tf.split(self.batch[5], self.num_gpus, name="lang_id_splits")
        tensors_input.shape_batch = tf.shape(self.batch[1])

        return tensors_input

    def build_tf_input_multilabel(self):
        """
        stand training input
        """
        tensors_input = namedtuple('tensors_input',
            'id_splits, feature_splits, label_splits, aux_label_splits, len_fea_splits, len_label_splits, len_aux_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                # split input data alone batch axis to gpus
                tensors_input.id_splits = tf.split(self.batch[0], self.num_gpus, name="id_splits")
                tensors_input.feature_splits = tf.split(self.batch[1], self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(self.batch[2], self.num_gpus, name="label_splits")
                tensors_input.aux_label_splits = tf.split(self.batch[3], self.num_gpus, name="aux_label_splits")
                tensors_input.len_fea_splits = tf.split(self.batch[4], self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(self.batch[5], self.num_gpus, name="len_label_splits")
                tensors_input.len_aux_label_splits = tf.split(self.batch[6], self.num_gpus, name='len_aux_label_splits')
        tensors_input.shape_batch = tf.shape(self.batch[1])

        return tensors_input

    def build_tf_input_kdbert(self):
        """
        stand training input
        """
        tensors_input = namedtuple('tensors_input',
                                   'id_splits, feature_splits, label_splits, bert_feat_splits, len_fea_splits, len_label_splits, len_bert_feat_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                # split input data alone batch axis to gpus
                tensors_input.id_splits = tf.split(self.batch[0], self.num_gpus, name="id_splits")
                tensors_input.feature_splits = tf.split(self.batch[1], self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(self.batch[2], self.num_gpus, name="label_splits")
                tensors_input.bert_feat_splits = tf.split(self.batch[3], self.num_gpus, name="bert_feat_splitss")
                tensors_input.len_fea_splits = tf.split(self.batch[4], self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(self.batch[5], self.num_gpus, name="len_label_splits")
                tensors_input.len_bert_feat_splits = tf.split(self.batch[6], self.num_gpus, name='len_bert_feat_splits')
        tensors_input.shape_batch = tf.shape(self.batch[1])

        return tensors_input

    def build_tf_input_ctcbert(self):
        """
                stand training input
                """
        tensors_input = namedtuple('tensors_input',
                                   'id_splits, feature_splits, label_splits, aux_label_splits, bert_feat_splits, len_fea_splits, len_label_splits, len_aux_label_splits, len_bert_feat_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                # split input data alone batch axis to gpus
                tensors_input.id_splits = tf.split(self.batch[0], self.num_gpus, name="id_splits")
                tensors_input.feature_splits = tf.split(self.batch[1], self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(self.batch[2], self.num_gpus, name="label_splits")
                tensors_input.aux_label_splits = tf.split(self.batch[3], self.num_gpus, name="aux_label_splits")
                tensors_input.bert_feat_splits = tf.split(self.batch[4], self.num_gpus, name="bert_feat_splits")
                tensors_input.len_fea_splits = tf.split(self.batch[5], self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(self.batch[6], self.num_gpus, name="len_label_splits")
                tensors_input.len_aux_label_splits = tf.split(self.batch[7], self.num_gpus, name='len_aux_label_splits')
                tensors_input.len_bert_feat_splits = tf.split(self.batch[8], self.num_gpus, name='len_bert_feat_splits')

        tensors_input.shape_batch = tf.shape(self.batch[1])

        return tensors_input

    def build_tf_input_ctcbertfull(self):
        """
                stand training input
                """
        tensors_input = namedtuple('tensors_input',
                                   'id_splits, feature_splits, label_splits, aux_label_splits, bertfull_feat_splits, len_fea_splits, len_label_splits, len_aux_label_splits, len_bertfull_feat_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                # split input data alone batch axis to gpus
                tensors_input.id_splits = tf.split(self.batch[0], self.num_gpus, name="id_splits")
                tensors_input.feature_splits = tf.split(self.batch[1], self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(self.batch[2], self.num_gpus, name="label_splits")
                tensors_input.aux_label_splits = tf.split(self.batch[3], self.num_gpus, name="aux_label_splits")
                tensors_input.bertfull_feat_splits = tf.split(self.batch[4], self.num_gpus, name="bertfull_feat_splits")
                tensors_input.len_fea_splits = tf.split(self.batch[5], self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(self.batch[6], self.num_gpus, name="len_label_splits")
                tensors_input.len_aux_label_splits = tf.split(self.batch[7], self.num_gpus, name='len_aux_label_splits')
                tensors_input.len_bertfull_feat_splits = tf.split(self.batch[8], self.num_gpus, name='len_bertfull_feat_splits')
        tensors_input.shape_batch = tf.shape(self.batch[1])

        return tensors_input

    def build_tf_input_multilingual(self):
        """
               stand training input
               """
        tensors_input = namedtuple('tensors_input',
                                   'id_splits, feature_splits, label_splits, lang_label_splits, len_fea_splits, len_label_splits, len_lang_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                # split input data alone batch axis to gpus
                tensors_input.id_splits = tf.split(self.batch[0], self.num_gpus, name="id_splits")
                tensors_input.feature_splits = tf.split(self.batch[1], self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(self.batch[2], self.num_gpus, name="label_splits")
                tensors_input.lang_label_splits = tf.split(self.batch[3], self.num_gpus, name="lang_label_splits")
                tensors_input.len_fea_splits = tf.split(self.batch[4], self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(self.batch[5], self.num_gpus, name="len_label_splits")
                tensors_input.len_lang_label_splits = tf.split(self.batch[6], self.num_gpus, name='len_lang_label_splits')
        tensors_input.shape_batch = tf.shape(self.batch[1])

        return tensors_input

    def build_tf_input_multilingual_multilabel(self):
        """
               stand training input
               """
        tensors_input = namedtuple('tensors_input',
                                   'id_splits, feature_splits, label_splits, aux_label_splits, lang_label_splits, len_fea_splits, len_label_splits, len_aux_label_splits, len_lang_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                # split input data alone batch axis to gpus
                tensors_input.id_splits = tf.split(self.batch[0], self.num_gpus, name="id_splits")
                tensors_input.feature_splits = tf.split(self.batch[1], self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(self.batch[2], self.num_gpus, name="label_splits")
                tensors_input.aux_label_splits = tf.split(self.batch[3], self.num_gpus, name="aux_label_splits")
                tensors_input.lang_label_splits = tf.split(self.batch[4], self.num_gpus, name="lang_label_splits")
                tensors_input.len_fea_splits = tf.split(self.batch[5], self.num_gpus, name="len_fea_splits")
                tensors_input.len_label_splits = tf.split(self.batch[6], self.num_gpus, name="len_label_splits")
                tensors_input.len_aux_label_splits = tf.split(self.batch[7], self.num_gpus, name="len_aux_label_splits")
                tensors_input.len_lang_label_splits = tf.split(self.batch[8], self.num_gpus, name='len_lang_label_splits')
        tensors_input.shape_batch = tf.shape(self.batch[1])

        return tensors_input

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        """
        be used for build infer model and the train model, conditioned on self.is_train
        """
        # build model in one device
        num_cell_units = self.args.model.num_cell_units
        cell_type = self.args.model.cell_type
        dropout = self.args.model.dropout
        forget_bias = self.args.model.forget_bias
        use_residual = self.args.model.use_residual

        hidden_output = tensors_input.feature_splits[id_gpu]
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            for i in range(self.args.model.num_lstm_layers):
                # build one layer: build block, connect block
                single_cell = build_cell(
                    num_units=num_cell_units,
                    num_layers=1,
                    is_train=self.is_train,
                    cell_type=cell_type,
                    dropout=dropout,
                    forget_bias=forget_bias,
                    use_residual=use_residual)
                hidden_output, _ = cell_forward(
                    cell=single_cell,
                    inputs=hidden_output,
                    index_layer=i)
                hidden_output = fully_connected(
                    inputs=hidden_output,
                    num_outputs=num_cell_units,
                    activation_fn=tf.nn.tanh,
                    scope='wx_b'+str(i))
                if self.args.model.use_layernorm:
                    hidden_output = layer_norm(hidden_output)

            logits = fully_connected(inputs=hidden_output,
                                     num_outputs=self.args.dim_output,
                                     activation_fn=tf.identity,
                                     scope='fully_connected')

            # Accuracy
            with tf.name_scope("label_accuracy"):
                correct = tf.nn.in_top_k(logits, tf.reshape(tensors_input.label_splits[id_gpu], [-1]), 1)
                correct = tf.multiply(tf.cast(correct, tf.float32), tf.reshape(tensors_input.mask_splits[id_gpu], [-1]))
                label_accuracy = tf.reduce_sum(correct)

            # Cross entropy loss
            with tf.name_scope("CE_loss"):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.reshape(tensors_input.label_splits[id_gpu], [-1]),
                    logits=logits)
                cross_entropy = tf.multiply(cross_entropy, tf.reshape(tensors_input.mask_splits[id_gpu], [-1]))
                cross_entropy_loss = tf.reduce_sum(cross_entropy)
                loss = cross_entropy_loss

            if self.is_train:
                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss)

        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Instances))

        return loss, gradients if self.is_train else logits, None

    def build_optimizer(self):
        if self.args.lr_type == 'stepped_down_decay':
            self.learning_rate = stepped_down_decay(
                self.global_step,
                learning_rate=self.args.learning_rate,
                decay_rate=self.args.decay_rate,
                decay_steps=self.args.decay_steps)
        elif self.args.lr_type == 'lr_decay_with_warmup':
            self.learning_rate = lr_decay_with_warmup(
                self.global_step,
                warmup_steps=self.args.warmup_steps,
                hidden_units=self.args.model.encoder.num_cell_units)
        elif self.args.lr_type == 'constant_learning_rate':
            self.learning_rate = tf.convert_to_tensor(self.args.constant_learning_rate)
        elif self.args.lr_type == 'exponential_decay':
            self.learning_rate = exponential_decay(
                self.global_step,
                lr_init=self.args.lr_init,
                lr_final=self.args.lr_final,
                decay_rate=self.args.decay_rate,
                decay_steps=self.args.decay_steps)
        else:
            self.learning_rate = warmup_exponential_decay(
                self.global_step,
                warmup_steps=self.args.warmup_steps,
                peak=self.args.peak,
                decay_rate=0.5,
                decay_steps=self.args.decay_steps)

        # if 'horovod' in sys.modules:
        if self.args.use_horovod:
            import horovod.tensorflow as hvd
            logging.info('wrap the optimizer with horovod!')
            # self.learning_rate = self.learning_rate * hvd.size()
            self.learning_rate = self.learning_rate

        with tf.name_scope("optimizer"):
            if self.args.optimizer == "adam":
                logging.info("Using ADAM as optimizer")
                optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                                   beta1=0.9,
                                                   beta2=0.98,
                                                   epsilon=1e-9,
                                                   name=self.args.optimizer)
            elif self.args.optimizer == "adagrad":
                logging.info("Using Adagrad as optimizer")
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            else:
                logging.info("Using SGD as optimizer")
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate,
                                                              name=self.args.optimizer)
        return optimizer

    def variables(self, scope=None):
        '''get a list of the models's variables'''
        scope = scope if scope else self.name
        scope += '/'
        print('all the variables in the scope:', scope)
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=scope)

        return variables


if __name__ == '__main__':

    from st.layers.arguments import args
    from dataProcessing import tfRecoderData
    import os

    logging.info('args.dim_input : {}'.format(args.dim_input))

    dataReader_train = tfRecoderData.TFRecordReader(args.dir_train_data, args=args)
    dataReader_dev = tfRecoderData.TFRecordReader(args.dir_dev_data, args=args)

    seq_features, seq_labels = dataReader_train.create_seq_tensor(is_train=False)
    batch_train = dataReader_train.fentch_batch_with_TFbuckets([seq_features, seq_labels], args=args)

    seq_features, seq_labels = dataReader_dev.create_seq_tensor(is_train=False)
    batch_dev = dataReader_dev.fentch_batch_with_TFbuckets([seq_features, seq_labels], args=args)

    tensor_global_step = tf.train.get_or_create_global_step()

    graph_train = LSTM_Model(batch_train, tensor_global_step, True, args)
    logging.info('build training graph successfully!')
    graph_dev = LSTM_Model(batch_dev, tensor_global_step, False, args)
    logging.info('build dev graph successfully!')

    writer = tf.summary.FileWriter(os.path.join(args.model_dir, 'log'), graph=tf.get_default_graph())
    writer.close()
    sys.exit()

    if args.is_debug:
        list_ops = [op.name+' '+op.device for op in tf.get_default_graph().get_operations()]
        list_variables_and_devices = [op.name+' '+op.device for op in tf.get_default_graph().get_operations() if op.type.startswith('Variable')]
        logging.info('\n'.join(list_variables_and_devices))

    list_kaldi_layers = []
    list_kaldi_layers = build_kaldi_lstm_layers(list_kaldi_layers, args.num_lstm_layers, args.dim_input, args.num_projs)
    list_kaldi_layers = build_kaldi_output_affine(list_kaldi_layers)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.train.MonitoredTrainingSession(config=config) as sess:
        kaldi_model = KaldiModel(list_kaldi_layers)
        kaldi_model.loadModel(sess=sess, model_path=args.model_init)
