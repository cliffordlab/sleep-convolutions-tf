
from __future__ import print_function
from __future__ import division

import os
import glob
# import logging
import numpy as np

# logger = logging.getLogger(name="model")

# depends: tf
import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Model, Sequential
from keras.layers.merge import concatenate
from tensorflow.python.estimator.model_fn import ModeKeys as Modes
from keras.engine.topology import Layer
from keras.layers import Reshape, Flatten, Dropout, Dense, Input
from keras.layers import Conv1D, Conv2D, MaxPooling1D, Activation
from keras.layers.normalization import BatchNormalization

from . import data as D
from . import tools as tl

class Scale(Layer):
    """rescales the input by a scalar tensor `Scale.W`"""

    add_summaries = True

    def __init__(self, init_val=1.0, **kwargs):
        super(Scale, self).__init__()
        self.init_val = init_val

    def build(self, input_shape):
        self.W = K.variable(self.init_val, name='{}_scale'.format(self.name))
        if self.add_summaries:
            tf.summary.scalar('Scale', self.W)
        self.trainable_weights = [self.W]
        super(Scale, self).build(input_shape)

    def call(self, x, mask=None):
        return self.W * x

    def compute_output_shape(self, input_shape):
        return input_shape


def Test_pipe(len_sequence=None, scale=1,
        num_layers=8, num_filters=8, kernel_size=15, kernel_factor=-0.1,
        filt_factor=0.1, pool_every_n=2, pool_size=3, dropout=0.0,
        batchnorm=0, scale_init=1.0):
    """Build configurable pipe for one channel with `sequence_len`"""
    assert pool_every_n > 0
    assert num_layers > 0
    model = Sequential()
    model.add(Reshape(input_shape=(len_sequence,), target_shape=(len_sequence, 1)))
    # Add scaling layer
    if scale:
        model.add(Scale(init_val=scale_init))
    for l in range(num_layers):
        # Add convolutional layer
        filters = int(num_filters*(1+filt_factor*l))
        kernels = int(kernel_size*(1+kernel_factor*l))
        if filters < 1: filters = 1
        if kernels < 1: kernels = 1
        model.add(Conv1D(
            filters = filters,
            kernel_size = kernels,
            padding = 'same'
        ))
        if batchnorm:
            model.add(BatchNormalization())
        model.add(Activation('relu'))
        # Maybe add pooling layer
        apply_pooling = ((l+1) % pool_every_n) == 0 and pool_size > 1
        if apply_pooling:
            stride = int(np.ceil(pool_size/2))
            model.add(MaxPooling1D(
                pool_size = pool_size,
                strides = stride,
                padding='same'
            ))
        if dropout > 0.0:
            model.add(Dropout(rate=dropout))
    return model


pipe_dict = {
    'test': Test_pipe,
    'EEG': Test_pipe,
    'EMG': Test_pipe,
    'EOG': Test_pipe,
}


def end_pipe_layers(num_dense_end_pipe=100, num_filters_end_pipe=24,
        kernel_size_end_pipe=5, dropout_end_pipe=0.5):
    layers = []
    layers.append(Conv2D(filters=num_filters_end_pipe, kernel_size=(4, kernel_size_end_pipe), padding='valid', activation='relu'))
    layers.append(Flatten())
    layers.append(Dense(num_dense_end_pipe, activation='relu'))
    layers.append(Dropout(rate=dropout_end_pipe))
    layers.append(Dense(num_dense_end_pipe, activation='relu'))
    layers.append(Dropout(rate=dropout_end_pipe))
    layers.append(Dense(6, activation='linear'))
    layers.append(Activation('softmax'))
    return layers


def determine_channel_type(channel_name):
    channel_type = None
    for t in pipe_dict:
        if channel_name.startswith(t):
            channel_type = t
            break
    assert channel_type is not None, "Unknown channel type [%s]" % channel_name
    return channel_type


def build_model(inp, print_summary, **kwargs):
    """Build tf graph of model input-to-output"""
    num_chan = len(inp)
    tensor = next(inp.values().__iter__())
    len_sequence = int(tensor.get_shape()[1])
    # Get default params
    pipe_params = tl.get_default_kwargs(Test_pipe)
    end_pipe_params = tl.get_default_kwargs(end_pipe_layers)
    # .. update with kwargs
    for k in pipe_params:
        pipe_params[k] = kwargs.get(k, pipe_params[k])
    pipe_params['len_sequence'] = len_sequence
    for k in end_pipe_params:
        end_pipe_params[k] = kwargs.get(k, end_pipe_params[k])

    # Pipes for each channel
    with tf.variable_scope("EEG-Pipe"):
        eeg_pipe = pipe_dict['EEG'](**pipe_params)
        if print_summary:
            print()
            eeg_pipe.summary()
        with tf.variable_scope("Ch1"):
            EEG1 = eeg_pipe(inp['EEG1'])
        with tf.variable_scope("Ch2"):
            EEG2 = eeg_pipe(inp['EEG2'])

    with tf.variable_scope("EMG-Pipe"):
        emg_pipe = pipe_dict['EMG'](**pipe_params)
        EMG1 = emg_pipe(inp['EMG1'])

    with tf.variable_scope("EOG-Pipe"):
        eog_pipe = pipe_dict['EOG'](**pipe_params)
        EOG1 = eog_pipe(inp['EOG1'])

    # Join channel-pipes
    # I wish i could to_join = [EEG1[:, :, None, :], EEG2[:, :, None, :], EMG1[:, :, None, :], EOG1[:, :, None, :]]
    output_sequence_length = int(EEG1.get_shape()[1])
    output_filters = int(EEG1.get_shape()[2])
    with tf.variable_scope('Join'):
        rshp = Reshape(target_shape=(1, output_sequence_length, output_filters))
        to_join = [rshp(t) for t in [EEG1, EEG2, EOG1, EMG1]]
        joined = concatenate(to_join, axis=1, name='Concat')

    with tf.variable_scope("End-Pipe"):
        end_layers = end_pipe_layers(**end_pipe_params)
        try:
            for l in end_layers[:-1]:
                joined = l(joined)
        except:
            exit(0)
        logits = joined

    with tf.variable_scope("Probs"):
        softm = end_layers[-1](logits)

    model = Model(
        outputs = softm,
        inputs = [inp[ch] for ch in D.channels],
        name = 'Model'
    )
    if print_summary:
        print()
        model.summary()
    return model, {'logits': logits, 'softmax': softm}


summary_printed = False

def model_fn(features, labels, mode, params):
    """Create tf graph for training, evaluation, or prediction
    features: dict of tensors
    labels: target tensor
    mode: Modes.TRAIN, .EVAL, .PREDICT ('infer')
    """
    global summary_printed
    # logger.debug("features={}, labels={}, mode={}, params={}".format(features, labels, mode, params))
    
    # Definitions
    EstimatorSpec = tf.estimator.EstimatorSpec

    # Configure model
    # Must set `K.backend.learning_phase` before building model
    if mode == Modes.TRAIN:
        K.set_learning_phase(1)
    elif mode in (Modes.EVAL, Modes.PREDICT):
        K.set_learning_phase(0)
    else:
        raise ValueError("Unknown mode {}".format(mode))

    features = {k: Input(tensor=T) for k, T in features.items()}
    # Full model (w/ 4 channels)
    defaults = tl.get_default_kwargs(Test_pipe, end_pipe_layers)
    model_params = tl.get_params_or_defaults(params, defaults)

    print_summary = not summary_printed
    model, outp = build_model(features, print_summary, **model_params)

    # Print model params once
    if not summary_printed:
        for k, v in model_params.items():
            print(k, v)
        summary_printed = True

    # Output mode specific
    if mode in (Modes.PREDICT, Modes.EVAL):
        with tf.variable_scope("Prediction"):
            predictions = tf.argmax(outp['softmax'], axis=-1)
        pred = {
            'probs': outp['softmax'],
            'classes': predictions
        }

    if mode in (Modes.TRAIN, Modes.EVAL):
        with tf.variable_scope("Loss"):
            loss = tf.losses.softmax_cross_entropy(labels, logits=outp['logits'])
            tf.summary.scalar('loss', loss)

    # Return values
    if Modes.PREDICT == mode:
        exp_outp = {
          'prediction': tf.estimator.export.PredictOutput(pred)
        }
        return EstimatorSpec(mode, predictions=predictions, export_outputs=exp_outp)

    if Modes.TRAIN == mode:
        # add training op
        with tf.variable_scope("Train"):
            Optimizer = tl.optimizer[params['opt']]
            opt_op = Optimizer(learning_rate=params['learning_rate'])
            train_op = opt_op.minimize(loss=loss, global_step=tf.train.get_global_step())
        # add summary ops
        with tf.name_scope("Summaries"):
            weights = {w.name: w for w in model.trainable_weights if 'conv' in
                    w.name}
            gradients = {'grad_'+k: tf.gradients(loss, v)[0] for k, v in
                    weights.items()}
            if 'weights' in params['summary']:
                for k, v in weights.items():
                    tf.summary.histogram(k[:-2], v)
            if 'gradients' in params['summary']:
                for k, v in gradients.items():
                    tf.summary.histogram(k[:-2], v)
        saver = tf.train.SummarySaverHook(save_steps=100,
                output_dir=params['model_dir'],
                summary_op=tf.summary.merge_all())
        return EstimatorSpec(mode, loss=loss, train_op=train_op,
                training_hooks=(saver,))

    if Modes.EVAL == mode:
        with tf.variable_scope("Metrics"):
            max_labels = tf.argmax(labels, -1)
            acc = tf.metrics.accuracy(max_labels, predictions)
            per_class_acc = tf.metrics.mean_per_class_accuracy(
                    max_labels, predictions, 6)
        eval_metric_ops = {
            'acc':  acc,
            'pca':  per_class_acc,
        }
        # funnel training/hptuning/metric through this iffy interface
        if tl.TfConfig().hptuning:
            eval_metric_ops['training/hptuning/metric'] = per_class_acc

        return EstimatorSpec(
                mode, loss=loss, predictions=predictions, eval_metric_ops=eval_metric_ops)


def build_estimator(config, **model_params):
    return tf.estimator.Estimator(config=config, model_fn=model_fn, params=model_params)


# Saver class (here unused)
class Saver(tf.train.Saver):
    """Saver combines tf.train.Saver and tf.summary.FileWriter"""

    # logger = logging.getLogger(name='Saver')
    logdir = './logs'
    suffix = 'model'
    ext = 'ckpt'

    def __init__(self, session, trial=0, **kwargs):
        super(Saver, self).__init__(*args, **kwargs)
        self.session = session
        self.trial = trial
        self.ckpt = self.newest_ckpt(trial)
        self.writer = tf.summary.FileWriter(self.model_dir(trial),
                graph=session.graph)

    def checkpoint(self, trial):
        """return all model checkpoints in <Saver.logdir>/trial"""
        from collections import defaultdict
        checkpoint = defaultdict(list)
        ckpt_file = os.path.join(self.logdir, str(trial), 'checkpoint') 
        if os.path.exists(ckpt_file):
            with open(ckpt_file, 'r') as f:
                for line in f:
                    key, rest = line.split(': "')
                    value = rest.split('"')[0]
                    checkpoint[key].append(value)
        return checkpoint['all_model_checkpoint_paths']

    def model_dir(self, trial):
        """return Saver.logdir/trial"""
        dn = os.path.join(self.logdir, str(trial))
        return dn

    def model_filename(self, trial, ckpt, check_exists=None):
        """return <Saver.logdir>/trial/<Saver.suffix>.ckpt.<Saver.ext>"""
        fn = os.path.join(self.model_dir(trial), '.'.join([self.suffix, str(ckpt), self.ext]))
        if check_exists is not None:
            if check_exists:
                assert os.path.exists(fn), '%s non-existent' % fn
            else:
                assert not os.path.exists(fn), '%s existent' % fn
        return fn

    def newest_ckpt(self, trial):
        """return newest ckpt in <Saver.logdir>/trial"""
        ckpt_names = self.checkpoint(trial)
        if len(ckpt_names) == 0:
            return 0
        else:
            ckpts = [int(f.split('.')[-2]) for f in ckpt_names]
            return np.max(ckpts)

    def from_checkpoint(self, fn):
        """restore session from checkpoint `fn`"""
        # self.logger.debug('Loading from %s' % fn)
        self.restore(self.session, fn)

    def from_newest_checkpoint(self, trial):
        """restore session from newest checkpoint in `trial`"""
        fn = self.model_filename(trial, self.newest_ckpt(trial), check_exists=True)
        # self.logger.debug('Loading from %s' % fn)
        self.restore(self.session, fn)

    def save_next(self, summary=None):
        """saves checkpoint to next ckpt-index, and writes summaries"""
        next_ckpt = 1+self.newest_ckpt(self.trial)
        fn = self.model_filename(self.trial, next_ckpt, check_exists=False)
        # self.logger.debug('Saving to %s' % fn)
        self.save(self.session, fn)
        if summary is not None:
            self.writer.add_summary(summary, next_ckpt)


# Patch keras.models.Model.summary
def _layer_connections(layer, relevant_nodes=None):
    connections = []
    for node_index, node in enumerate(layer.inbound_nodes):
        if relevant_nodes:
            node_key = layer.name + '_ib-' + str(node_index)
            if node_key not in relevant_nodes:
                # node is node part of the current network
                continue
        for i in range(len(node.inbound_layers)):
            inbound_layer = node.inbound_layers[i].name
            inbound_node_index = node.node_indices[i]
            inbound_tensor_index = node.tensor_indices[i]
            connections.append(inbound_layer + '[' + str(inbound_node_index) + '][' + str(inbound_tensor_index) + ']')
    return connections


def _format_output_shape(layer):
    try:
        output_shape = layer.output_shape
        if output_shape[0] is None:
            output_shape = ('?',) + output_shape[1:]
        output_shape = 'x'.join(map(str, output_shape))
        return output_shape
    except:
        return 'multiple'


def save_summary(model, filename):
    layers = model.layers
    relevant_nodes = model.container_nodes if hasattr(model, 'container_nodes') else None
    column_names = ['Layer', 'Type', 'Output Shape', 'Param #', 'Connected to']

    summary = []
    for layer in layers:
        summary.append([
            layer.name,
            layer.__class__.__name__,
            _format_output_shape(layer),
            layer.count_params(),
            '\n'.join(_layer_connections(layer, relevant_nodes=relevant_nodes))
        ])

    import pandas
    df = pandas.DataFrame(data=summary, columns=column_names)    
    df.to_csv(filename, index=False)

Model.save_summary = save_summary


def ckpt_to_keras(checkpoint, modelfile, **params):
    import h5py
    import keras
    import keras.backend as K
    seq_len = int(D.sr*D.dt)
    params = {k.replace('-', '_'): v for k, v in params.items()}
    with tf.Graph().as_default():
        features = {
            tp: keras.Input(shape=(seq_len,), name=tp)
            for tp in D.channels
        }
        model, _ = build_model(features, print_summary=True, **params)
        saver = tf.train.Saver()
        with tf.Session() as S:
            saver.restore(S, checkpoint)
            K.manual_variable_initialization(True)
            model.save(modelfile)
