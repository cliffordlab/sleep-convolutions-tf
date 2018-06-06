
from __future__ import print_function
from __future__ import division

import os
# import logging
import pickle
import numpy as np
from subprocess import check_output
from .tools import gs, tf_scope, fft_surrogate
# logger = logging.getLogger(name='data')

# Data constants
events = 'Wake S1 S2 S3 S4 REM'.split()
channels = 'EEG1 EEG2 EMG1 EOG1'.split()
dt = 30.0  # sec.
sr = 32.0  # Hz
LEN_SEQUENCE = int(dt*sr)

color = {
    events[0]: '#FF4848',
    events[1]: '#72FB72',
    events[2]: '#006400',
    events[3]: '#ADD8E6',
    events[4]: '#033591',
    events[5]: '#FFBF48',
}

age_group_bins = {
    'bin1': '13-28',
    'bin2': '28-42',
    'bin3': '42-55',
    'bin4': '55-69',
    'bin5': '69-82'
}

# multilabel encoding
_encoder = {evt: i for i, evt in enumerate(events)}
_decoder = {v: k for k, v in _encoder.items()}
encode = np.vectorize(_encoder.get)
decode = np.vectorize(_decoder.get)


def tmp_dir(resource_id=None):
    if resource_id is None:
        import uuid
        resource_id = str(uuid.uuid4())
    d = os.path.join('/', 'tmp', 'jusjusjus', resource_id)
    if not os.path.exists(d):
        os.makedirs(d)
    return d


def copy_if_remote(filename):
    """If on gs bucket, copy file to tmp-dir. Returns new filename"""
    import os
    import subprocess
    if gs.its_remote(filename):
        basename = os.path.basename(filename)
        localfile = os.path.join(tmp_dir(), basename)

        # gs.cp creates dirs that don't exist yet.
        gs.cp(filename, localfile)
        filename = localfile

    return filename


def load_data(filename):
    """Load pickled or zipped dict of features and target data"""
    filename = copy_if_remote(filename)
    assert isinstance(filename, str), "load_data(filename) got {} as argument (requires {})".format(filename, str)
    _, ext = os.path.splitext(filename)
    if ext == '.pkl':
        with open(filename, 'rb') as f:
            data = pickle.load(f)
    elif ext == '.npz' or ext == '.npy': 
        data = np.load(filename)
    else:
        raise ValueError("unknown extension [{}]".format(filename))
    return data


# TFRecords part
def build_tf_feature_map(len_sequence=None):
    import tensorflow as tf
    shp = () if len_sequence is None else (len_sequence,)
    fmap = {
        ch: tf.FixedLenSequenceFeature(shape=shp, dtype=tf.float32,
            allow_missing=True)
        for ch in channels
    }
    fmap['target'] = tf.FixedLenSequenceFeature(shape=(),
            dtype=tf.int64, allow_missing=True)
    return fmap


def read_tfrecords(f):
    """Returns dictionary with all entries of f."""
    # global local_init_op
    import tensorflow as tf
    from collections import defaultdict
    with tf.Graph().as_default():
        f_Q = tf.train.string_input_producer([f], num_epochs=1, shuffle=False)
        _, serialized = tf.TFRecordReader().read_up_to(f_Q, 1024)
        examples = tf.parse_example(serialized, features=build_tf_feature_map())
        examples['target'] = tf.squeeze(examples['target'], axis=1)
        with tf.Session() as S:
            S.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            read_examples = defaultdict(list)
            while True:
                try:
                    new_examples = S.run(examples)
                    for k, v in new_examples.items():
                        read_examples[k].append(v)
                except:
                    coord.request_stop()
                    coord.join(threads)
                    break
    return {k: np.concatenate(v) for k, v in read_examples.items()}


class TFRecordFile(object):

    columns = channels + ['target']

    def __init__(self, filename, cleanup=False):
        self.cleanup = cleanup
        self.filename = filename
        self.read()

    def data_cached(self):
        return all(
            os.path.exists(f)
            for f in (self.features_filename, self.target_filename)
        )

    def read(self):
        if not self.data_cached():
            data = read_tfrecords(self.filename)
            assert all(c in data for c in self.columns), "{} missing column [{}]".format(self.filename, data)
            self.features = np.array([data[c] for c in channels], dtype=np.float32)
            self.target = data['target']

    @property
    def num_samples(self):
        return self.features_shape[1]

    @property
    def features(self):
        features = np.memmap(self.features_filename, mode='r', dtype=np.float32,
                shape=self.features_shape)
        return features

    @features.setter
    def features(self, x):
        self.features_shape = x.shape
        features = np.memmap(self.features_filename, mode='w+', dtype=np.float32,
                shape=self.features_shape)
        features[:] = x[:]
        del features

    @property
    def target(self):
        target = np.memmap(self.target_filename, mode='r', dtype=np.int,
                shape=(self.features_shape[1],))
        return target

    @target.setter
    def target(self, x):
        self.target_filename = self.tmpfile('target') 
        target = np.memmap(self.target_filename, mode='w+', dtype=np.int, shape=x.shape)
        target[:] = x[:]
        del target

    @property
    def features_shape(self):
        if self._features_shape is None:
            self._features_shape = tuple(np.load(self.tmpfile('shape')))
        return self._features_shape

    @features_shape.setter
    def features_shape(self, shp):
        self._features_shape = shp
        np.save(self.tmpfile('shape'), shp)

    @property 
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, fn):
        if not gs.its_remote(fn):
            assert os.path.exists(fn), '{} not existent'.format(fn)
        self._filename = fn
        abs_dir_parts = os.path.dirname(os.path.abspath(fn)).split('/')
        self.dir = os.path.join(*abs_dir_parts[3:])
        self.base, _ = os.path.splitext(os.path.basename(fn))
        self.features_filename = self.tmpfile('features') 
        self.target_filename = self.tmpfile('target') 
        self._features_shape = None

    def tmpfile(self, name):
        rc = os.path.join(self.dir, self.base)
        return os.path.join(tmp_dir(resource_id=rc), name+'.npy')

    def __del__(self):
        if self.cleanup:
            for fname in (self.features_filename, self.target_filename):
                if fname is not None and os.path.exists(fname):
                    os.remove(fname)


class TFRecordDataset:

    augment_fn = fft_surrogate

    def __init__(self, filenames, batch_size, train=True, balance_factor=1.0,
            augment=0.0):
        self.filenames = filenames
        self.batch_size = batch_size
        self.train = train
        self.balance_factor = balance_factor
        self.augment = augment

    def read(self):
        """Read all files and store them in memory.  This is a bad idea
        for memory consumption."""
        records = [TFRecordFile(f, cleanup=False) for f in self.filenames]
        num_channels, _, seq_length = records[0].features.shape
        idx_list = [0]+[r.num_samples for r in records]
        idx_cum = np.cumsum(idx_list)
        num_samples = self.num_samples = sum(idx_list)
        self.features_shape = (num_channels, num_samples, seq_length)

        # Parse all recordings, and generate a copy for mixing
        features = np.memmap(self.tmpfile('features'), mode='w+',
                dtype=np.float32, shape=self.features_shape)
        target = np.memmap(self.tmpfile('target'), mode='w+',
                dtype=np.int, shape=(num_samples,))
        for start, end, record in zip(idx_cum[:-1], idx_cum[1:], records):
            print('reading file', record.filename)
            f, t = record.features[:], record.target[:]
            features[:, start:end] = f 
            target[start:end] = t 
        del features, target

        if self.train:
            self.shuffle()


    def shuffle(self):
        batch_size = 4*1024
        num_samples = self.features_shape[1]
        batches = list(range(0, num_samples, batch_size))
        batches.append(num_samples)

        # copy features->features_copy
        ## read features and target
        features = np.memmap(self.tmpfile('features'), mode='r',
                dtype=np.float32, shape=self.features_shape)
        target = np.memmap(self.tmpfile('target'), mode='r',
                dtype=np.int, shape=(num_samples,))
        ## create arrays to hold the copy
        features_copy = np.memmap(self.tmpfile('features_copy'), mode='w+',
                dtype=np.float32, shape=self.features_shape)
        target_copy = np.memmap(self.tmpfile('target_copy'), mode='w+',
                dtype=np.int, shape=(num_samples,))
        for start, end in zip(batches[:-1], batches[1:]):
            s = slice(start, end)
            features_copy[:, s] = features[:, s]
            target_copy[s] = target[s]
        del features_copy, target_copy

        # write shuffle(features_copy) -> features
        permutation = np.random.permutation(num_samples)
        features = np.memmap(self.tmpfile('features'), mode='w+',
                dtype=np.float32, shape=self.features_shape)
        target = np.memmap(self.tmpfile('target'), mode='w+',
                dtype=np.int, shape=(num_samples,))
        features_copy = np.memmap(self.tmpfile('features_copy'), mode='r',
                dtype=np.float32, shape=self.features_shape)
        target_copy = np.memmap(self.tmpfile('target_copy'), mode='r',
                dtype=np.int, shape=(num_samples,))
        for start, end in zip(batches[:-1], batches[1:]):
            s = slice(start, end)
            ps = permutation[s]
            features[:, s] = features_copy[:, ps]
            target[s] = target_copy[ps]
        del features, target
        self.remove('features_copy')
        self.remove('target_copy')


    def balance(self):
        batch_size = 4*1024
        features = np.memmap(self.tmpfile('features'), mode='r',
                dtype=np.float32, shape=self.features_shape)
        target = np.memmap(self.tmpfile('target'), mode='r',
                dtype=np.int, shape=(self.features_shape[1],))

        num_channels, num_samples, seq_length = self.features_shape

        # determine number of repeats
        # including/not including the originals
        labels, cnts_per_lbl = np.unique(target, return_counts=True)
        max_cnts = np.max(cnts_per_lbl)
        missing_cnts_per_lbl = max_cnts-cnts_per_lbl
        additional_cnts = (self.balance_factor*missing_cnts_per_lbl).astype(int)
        repeats_per_lbl = np.ceil(additional_cnts/cnts_per_lbl).astype(int)
        new_cnts_per_lbl = cnts_per_lbl + additional_cnts

        # repeat an index
        Range = np.arange(num_samples)
        label_indices = [target==l for l in labels]
        repeated = np.concatenate([
            np.repeat(Range[idx], r)[:c]
            for idx, r, c in zip(
                label_indices, repeats_per_lbl, additional_cnts)
        ])
        original_and_repeated = np.concatenate([
            np.repeat(Range[idx], 1+r)[:c]
            for idx, r, c in zip(
                label_indices, repeats_per_lbl, new_cnts_per_lbl)
        ])
        # labels, cnts_per_lbl = np.unique(target[repeated], return_counts=True)
        # for l, c in zip(labels, cnts_per_lbl):
        #     print('repeats:', l, c)
        self.repeated_features_shape = (
                num_channels, num_samples+repeated.size, seq_length)

        # write the originals of size num_samples
        unshuffled_features = np.memmap(self.tmpfile('unshuffled_features'), mode='w+',
                dtype=np.float32, shape=self.repeated_features_shape)
        unshuffled_target = np.memmap(self.tmpfile('unshuffled_target'), mode='w+',
                dtype=np.int, shape=(self.repeated_features_shape[1],))
        batches = list(range(0, num_samples, batch_size))
        batches.append(num_samples)
        for start, end in zip(batches[:-1], batches[1:]):
            s = slice(start, end)
            unshuffled_features[:, s] = features[:, s]
            unshuffled_target[s] = target[s]
        # for further writes, num_samples is the offset
        offset = num_samples

        # write the repetitions (possibly augmented)
        batches = list(range(0, repeated.size, batch_size))
        batches.append(repeated.size)
        for start, end in zip(batches[:-1], batches[1:]):
            s = slice(start, end)
            s_offset = slice(offset+start, offset+end)
            prs = repeated[s]
            sliced_features = features[:, prs] 
            if self.augment > 0.0:
                sliced_features = self.augment_features(sliced_features)
            unshuffled_features[:, s_offset] = sliced_features
            unshuffled_target[s_offset] = target[prs]

        del unshuffled_features, unshuffled_target

        # Shuffle the samples and push to `repeated_features`
        shuffled = np.arange(num_samples+repeated.size)
        np.random.shuffle(shuffled)
        batches = list(range(0, shuffled.size, batch_size))
        batches.append(shuffled.size)
        slices = (slice(s, e) for s, e in zip(batches[:-1], batches[1:]))

        unshuffled_features = np.memmap(self.tmpfile('unshuffled_features'),
                mode='r', dtype=np.float32, shape=self.repeated_features_shape)
        unshuffled_target = np.memmap(self.tmpfile('unshuffled_target'),
                mode='r', dtype=np.int, shape=(self.repeated_features_shape[1],))
        repeated_features = np.memmap(self.tmpfile('repeated_features'),
                mode='w+', dtype=np.float32, shape=self.repeated_features_shape)
        repeated_target = np.memmap(self.tmpfile('repeated_target'),
                mode='w+', dtype=np.int, shape=(self.repeated_features_shape[1],))

        for s in slices:
            sh = shuffled[s]
            repeated_features[:, s] = unshuffled_features[:, sh]
            repeated_target[s] = unshuffled_target[sh]

        del repeated_features, repeated_target
        self.remove('features')
        self.remove('target')
        self.remove('unshuffled_features')
        self.remove('unshuffled_target')

    def augment_features(self, features):
        num_channels, num_examples, seq_len = features.shape
        augmented = np.empty_like(features)
        prob = self.augment > np.random.rand(num_channels, num_examples)
        for c in range(num_channels):
            for e in range(num_examples):
                if prob[c, e]:
                    augmented[c, e] = self.augment_fn(features[c, e])
                else:
                    augmented[c, e] = features[c, e]
        return augmented

    def remove(self, name):
        fname = self.tmpfile(name)
        if os.path.exists(fname):
            os.remove(fname)

    def __del__(self):
        for f in ('features', 'target', 'features_copy', 'target_copy',
                'repeated_features', 'repeated_target'):
            self.remove(f)

    def tmpfile(self, name):
        intention = 'train' if self.train else 'test'
        return os.path.join(tmp_dir(resource_id='dataset'), '.'.join([name, intention, 'npy']))

    def gen_input_fn(self):
        import tensorflow as tf

        if self.train:
            num_epochs = None
            self.balance()
            features_name = 'repeated_features'
            target_name = 'repeated_target'
            shp = self.repeated_features_shape
        else:
            num_epochs = 1
            features_name = 'features'
            target_name = 'target'
            shp = self.features_shape
        
        features = np.memmap(self.tmpfile(features_name), mode='r',
                dtype=np.float32, shape=shp)
        target = np.memmap(self.tmpfile(target_name), mode='r',
                dtype=np.int, shape=(shp[1],))

        features = {
            channel: features[c]
            for c, channel in enumerate(channels)
        }

        class input_fn:

            def __init__(self, features, target, batch_size, num_epochs):
                self.features = features
                self.target = target
                self.batch_size = batch_size
                self.num_epochs = num_epochs

            def __call__(self):
                features, target = tf.estimator.inputs.numpy_input_fn(
                    x = self.features,
                    y = self.target,
                    batch_size = self.batch_size,
                    num_epochs = num_epochs,
                    shuffle = False
                )()
                target = tf.one_hot(target, 6)
                return features, target
                

        return input_fn(features, target, self.batch_size, num_epochs)


def generate_tfrecords_input_fn(filenames, batch_size, train=True,
        balance_factor=1.0, channel_subset=None, augment=0.0):
    """Returns generator from tfrecords files"""
    dataset = TFRecordDataset(filenames, batch_size, train=train,
            balance_factor=balance_factor, augment=augment)
    dataset.read()
    return dataset.gen_input_fn()


GEN_INPUT_FUNCTIONS = {
    '.tfrecords': generate_tfrecords_input_fn,
}


# backwards compatibility
def generate_input_fn(filenames, *args, **kwargs):
    import os
    _, ext = os.path.splitext(filenames[0])
    assert ext in GEN_INPUT_FUNCTIONS, "extension not found [%s]"%filenames[0]
    return GEN_INPUT_FUNCTIONS[ext](filenames, *args, **kwargs)


def generate_tfrecords_serving_input_fn(channel_subset=None):

    def tfrecords_serving_input_fn():
        import tensorflow as tf
        seq_length = int(dt*sr) 
        examples = tf.placeholder(tf.string, shape=())
        feat_map = {
            channel: tf.FixedLenSequenceFeature(shape=(seq_length,),
                dtype=tf.float32, allow_missing=True)
            for channel in channels
        }
        parsed = tf.parse_single_example(examples, features=feat_map)
        if channel_subset is None:
            features = {
                channel: tf.expand_dims(tensor, -1)
                for channel, tensor in parsed.iteritems()
            }
        else:
            features = {
                c: tf.expand_dims(parsed[c], -1)
                for c in channel_subset
            }
        from collections import namedtuple
        InputFnOps = namedtuple("InputFnOps", "features labels receiver_tensors")
        tf.contrib.learn.utils.input_fn_utils.InputFnOps = InputFnOps
        return InputFnOps(features=features, labels=None, receiver_tensors=examples)
        # InputFnOps = tf.contrib.learn.utils.input_fn_utils.InputFnOps
        # return InputFnOps(features, None, {'features': parsed})
        # Error: InputFnOps has no attribute receiver_tensors

    return tfrecords_serving_input_fn


SERVING_FUNCTIONS = {
    'TF_RECORD': generate_tfrecords_serving_input_fn,
}
