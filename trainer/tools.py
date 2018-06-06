
import os
import numpy as np
from subprocess import check_output

def endless_cache(fn):
    cache = {}
    def wrapped(*args):
        try:
            rep = '_'.join([str(a) for a in args])
            res = cache[rep]
        except:
            res = fn(*args)
            cache[rep] = res
        finally:
            return res
    
    return wrapped


class gs:

    prefix = 'gs://'

    @classmethod
    def join(gs, *args, **kwargs):
        add_prefix = kwargs.get('add_prefix', True)
        parts = [p.strip('/') for p in args]
        if not gs.its_remote(parts[0]) and add_prefix:
            parts[0] = gs.prefix+parts[0]
        return '/'.join(parts)


    @classmethod
    def makedirs(gs, dirname):
        try:
            os.makedirs(dirname)
            print('created local directory', dirname)
        except:
            pass


    @classmethod
    def cp(gs, source, dest, *args):
        gs.assert_any_remote(source, dest)
        if not gs.its_remote(dest):
            gs.makedirs(os.path.dirname(dest))
        cmd_list = ('gsutil', 'cp') + args + (source, dest)
        res = check_output(cmd_list)
        assert os.path.exists(dest), "failed to copy file [{}]".format(cmd_list)
        return res

    @classmethod
    def its_remote(gs, x):
        return x.startswith(gs.prefix)

    @classmethod
    def assert_any_remote(gs, *args):
        assert any(gs.its_remote(f) for f in args), "All files local! {}".format(args)

    @classmethod
    def describe_hopt_job(gs, job_name):
        import subprocess
        import yaml
        output = subprocess.check_output(['gcloud', 'ml-engine', 'jobs', 'describe', job_name])
        return yaml.load(output)


def get_default_kwargs(*fns):
    """return dict of key-word arguments (for fn with kwargs only)"""
    import inspect
    kwargs = {}
    for fn in fns:
        spec = inspect.getargspec(fn)
        keys = spec.args
        vals = spec.defaults
        if len(keys) == 0 and vals is None:
            continue
        assert len(keys) == len(vals), 'Non keyword args detected [keys: {}, vals: {}]'.format(keys, vals)
        kwargs.update(dict(zip(keys, vals)))
    return kwargs


def pop_params_or_defaults(params, defaults):
    """Returns params.update(default) restricted to default keys"""
    merged = {
        key: params.pop(key, default)
        for key, default in defaults.items()
    }
    return merged


def get_params_or_defaults(params, defaults):
    """Returns params.update(default) restricted to default keys."""
    merged = {
        key: params.get(key, default)
        for key, default in defaults.items()
    }
    return merged


class TfConfig:

    def __init__(self):
        import os
        from collections import defaultdict
        self.TF_CONFIG = defaultdict(list)
        if 'TF_CONFIG' in os.environ:
            import json
            tfcf = json.loads(os.environ['TF_CONFIG'])
            self.TF_CONFIG.update(**tfcf)

    @property
    def hptuning(self):
        return 'hyperparameters' in self.TF_CONFIG['job']

        


import tensorflow as tf
optimizer = {
    'sgd': tf.train.GradientDescentOptimizer,
    'adagrad': tf.train.AdagradOptimizer,
    'adadelta': tf.train.AdadeltaOptimizer,
    'rmsprop': tf.train.RMSPropOptimizer,
}


def tf_scope(name):
    import tensorflow as tf
    def _tf_scope(fn):
        def wrapped(*args, **kwargs):
            with tf.variable_scope(name):
                return fn(*args, **kwargs)
        return wrapped
    return _tf_scope


def tf_device(name):
    import tensorflow as tf
    def _tf_scope(fn):
        def wrapped(*args, **kwargs):
            with tf.device(name):
                return fn(*args, **kwargs)
        return wrapped
    return _tf_scope


class HP:
    TYPE = {
        'INTEGER': int,
        'DOUBLE': float,
        'CATEGORICAL': list,
        'DISCRETE': list
    }
    SCALETYPE = {
        None: None,
        'UNIT_LOG_SCALE': 'log-uniform'
    }

    SCALETRAFO = {
        'log-uniform': (np.log10, lambda x: 10.0**x)
    }
    
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        t = self.type = self.TYPE[self.type]
        self.scaleType = self.SCALETYPE[getattr(self, 'scaleType', None)]
        if t is not list:
            proplist = ('minValue', 'maxValue')
        else:
            proplist = []
        for prop in proplist:
            try:
                setattr(self, prop, t(getattr(self, prop)))
            except:
                self.minValue = 0.0

        self.values = []
    
    @property
    def range(self):
        if self.type is list:
            raise AttributeError("HP has no range {}".format(self.__dict__))
        if self.scaleType is not None: 
            return (self.minValue, self.maxValue, self.scaleType)
        else:
            return (self.minValue, self.maxValue)

    def padded_range(self, pad):
        MIN, MAX = self.range[:2]
        pad = pad*(MAX-MIN)
        return (MIN-pad, MAX+pad)

    def append(self, x):
        self.values.append(self.type(x))
    
    def __str__(self):
        return str(self.range)


class HPJob:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.hyperparameters = {
            p['parameterName']: HP(**p)
            for p in self.trainingInput['hyperparameters']['params']
        }
        self.objective = []
        self.parsed = False
        self.parse_output()
        self._cutoff = None
        
    @classmethod
    def from_gcloud_job(cls, job_name):
        output_dict = gs.describe_hopt_job(job_name)
        return cls(**output_dict)
    
    def parse_output(self):
        if self.parsed:
            return
        for trial in self.trainingOutput['trials']:
            try:
                self.objective.append(trial['finalMetric']['objectiveValue'])
                for k in self.hyperparameters:
                    self.hyperparameters[k].append(trial['hyperparameters'][k])
            except:
                print('failed trial:', trial)
        self.parsed = True

    @property
    def cutoff(self):
        return self._cutoff

    @cutoff.setter
    def cutoff(self, c):
        assert 0.0 < c < 1.0
        self._cutoff = c
        
    @property
    def spaces(self):
        return {
            pn: hp.range
            for pn, hp in self.hyperparameters.items()
            if hp.type is not list
        }
    
    @property
    def trials_and_results(self):
        import numpy as np
        trials = np.array([
            hp.values
            for _, hp in self.hyperparameters.items()
            if hp.type is not list
        ]).transpose()
        trials, results = trials, np.array(self.objective)
        if self.cutoff is None:
            return trials, results
        else:
            idx = (objective > self.cutoff)
            return hyperpars[idx], objective[idx]


    def apply_cutoff(self, cutoff):
        hyperpars, objective = self.trials_and_results

    
    def plot_objective(self, filename=None, n_points=25, n_samples=250,
            cutoff=0.0):
        # if `savefig(filename)`, don't `show()`
        if filename is not None:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import skopt
        from skopt.plots import plot_objective

        spaces = list(self.spaces.values())
        pnames = list(self.spaces.keys())
        opt = skopt.Optimizer(spaces, "ET", acq_optimizer="sampling")
        self.cutoff = cutoff
        hyperpars, objective = self.trials_and_results
        # skopt minimizes.  Therefore use: acc = 1-acc
        objective = 1-objective
        hyperpars = [list(f) for f in hyperpars]
        objective = list(objective)
        opt.tell(hyperpars, objective)
        opt_result = opt.run(lambda x: 0, n_iter=0)
        plot_objective(opt_result, n_points=n_points, n_samples=n_samples, dimensions=pnames)
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    def plot_evaluations(self, filename=None, n_points=25, n_samples=250):
        # if `savefig(filename)`, don't `show()`
        if filename is not None:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import skopt
        from skopt.plots import plot_evaluations

        spaces = list(self.spaces.values())
        pnames = list(self.spaces.keys())
        opt = skopt.Optimizer(spaces, "ET", acq_optimizer="sampling")
        hyperpars, objective = self.trials_and_results
        # skopt minimizes.  Therefore use: acc = 1-acc
        objective = 1-objective
        hyperpars = [list(f) for f in hyperpars]
        objective = list(objective)
        opt.tell(hyperpars, objective)
        opt_result = opt.run(lambda x: 0, n_iter=0)
        plot_evaluations(opt_result, bins=20, dimensions=pnames)
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    def plot_convergence(self, filename=None, n_points=25, n_samples=250):
        # if `savefig(filename)`, don't `show()`
        if filename is not None:
            import matplotlib
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import skopt
        from skopt.plots import plot_convergence

        spaces = list(self.spaces.values())
        pnames = list(self.spaces.keys())
        opt = skopt.Optimizer(spaces, "ET", acq_optimizer="sampling")
        hyperpars, objective = self.trials_and_results
        # skopt minimizes.  Therefore use: acc = 1-acc
        objective = 1-objective
        hyperpars = [list(f) for f in hyperpars]
        objective = list(objective)
        opt.tell(hyperpars, objective)
        opt_result = opt.run(lambda x: 0, n_iter=0)
        plot_convergence(opt_result)#, n_points=n_points, n_samples=n_samples, dimensions=pnames)
        plt.tight_layout()
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            
            
def cosine_transition(transition, sr=None):
    if not isinstance(transition, int):
        transition = int(transition*sr)
    th = np.linspace(0., np.pi, num=transition, endpoint=np.pi)
    falling = (1+np.cos(th))/2
    rising = 1-falling
    return falling, rising

def cut_and_glue(x, start, end, transition, sr=None):
    if not isinstance(start, int):
        start = int(start*sr)
    if not isinstance(end, int):
        end = int(end*sr)
    falling, rising = cosine_transition(transition, sr)
    s = falling.size+start
    e = end-rising.size
    trans = x[start:s]*falling+x[e:end]*rising
    return np.concatenate([x[:start], trans, x[end:]])

def finite_support(duration, transition, sr=None):
    if not isinstance(duration, int):
        duration = int(duration*sr)
    if not isinstance(transition, int):
        transition = int(transition*sr)
    assert duration > 2*transition, "transition [{} pts] exceeds duration [{} pts]".format(transition, duration)
    falling, rising = cosine_transition(transition, sr=None)
    return np.concatenate([
        falling, np.zeros(duration-2*transition, dtype=np.float64), rising])

def partial_fft_surrogate(x, start, end, transition, sr=None, f=None):
    import scipy.fftpack as sft
    if not isinstance(start, int):
        start = int(start*sr)
    
    if not isinstance(end, int):
        end = int(end*sr)
    duration = end-start
    
    if f is None:
        glued = cut_and_glue(x, start, end, transition, sr=sr)
        f = sft.fft(glued.astype(np.float64))
    
    surrogate = fft_surrogate(f=f)
    support = finite_support(duration, transition, sr=sr)
    slices = [
        slice(duration*i, duration*(i+1))
        for i in range(surrogate.size//duration)
    ]
    replaced = x[start:end]*support
    partial_surrogates = np.repeat(
        x, len(slices)).reshape(x.size, len(slices)).transpose() 
    for b, s in enumerate(slices):
        partial_surrogates[b, start:end] = replaced+surrogate[s]*(1-support)
    
    return partial_surrogates

def new_random_fft_phase_odd(n):
    random_phase = 2j*np.pi*np.random.rand((n-1)//2)
    return np.concatenate([[0.0], random_phase, -random_phase[::-1]])

def new_random_fft_phase_even(n):
    random_phase = 2j*np.pi*np.random.rand(n//2-1)
    return np.concatenate([[0.0], random_phase, [0.0], -random_phase[::-1]])

new_random_fft_phase = {
    0: new_random_fft_phase_even,
    1: new_random_fft_phase_odd
}

def fft_surrogate(x=None, f=None):
    import scipy.fftpack as sft
    if f is None:
        assert x is not None, 'Neither x nor f provided.'
        f = sft.fft(x.astype(np.float64))
    n = f.size
    random_phase = new_random_fft_phase[n%2](n)
    f_shifted = f*np.exp(random_phase)
    shifted = sft.ifft(f_shifted)
    return shifted.astype(np.float64)


def partial_zero_out(x, start, end, transition, sr=None, f=None):
    if not isinstance(start, int):
        start = int(start*sr)
    
    if not isinstance(end, int):
        end = int(end*sr)
    duration = end-start
    
    support = finite_support(duration, transition, sr=sr)
    partial_zero = np.copy(x)
    partial_zero[start:end] = x[start:end]*support
    return partial_zero

def generate_zero_out_scan(
        X, stride=1, channels=None, width=2.5, transition=0.5, sr=None):
    from collections import namedtuple
    Segment = namedtuple('Segment', 'start end transition')
    if not isinstance(width, int):
        width = int(sr * width)
    width2 = width//2
    if not isinstance(transition, int):
        transition = int(sr * transition)
    num_channels, num_samples = X.shape
    if channels is None:
        channels = slice(0, num_channels)
    else:
        from . import data as D
        channels = [D.channels.index(c) for c in channels]

    slices = [
        Segment(start=max(0, n-width2), 
                end=min(num_samples, n+width2), 
                transition=transition)
        for n in range(0, num_samples, stride)
    ]
    repeats = len(slices)
    zero_outs = np.repeat(X, repeats).reshape(num_channels, -1, repeats).swapaxes(1, 2)
    for i, segment in enumerate(slices):
        zero_outs[channels, i] = np.array([
            partial_zero_out(x, *segment)
            for x in X[channels]
        ])
    return zero_outs


def partial_fft_surrogate_batch(x, batch_size, start, end, transition, sr=None, f=None):
    import scipy.fftpack as sft
    if f is None:
        glued = cut_and_glue(x, start, end, transition, sr=sr)
        f = sft.fft(glued.astype(np.float64))
    surrogates = partial_fft_surrogate(x, start, end, transition, sr=sr, f=f)
    while surrogates.shape[0] < batch_size:
        more_surrogates = partial_fft_surrogate(x, start, end, transition, sr=sr, f=f)
        surrogates = np.vstack([surrogates, more_surrogates])
    return surrogates[:batch_size]

def generate_partial_surrogate_batches(
        X, batch_size, stride=1, channels=None, width=2.5, transition=0.5, sr=None):
    from collections import namedtuple
    Segment = namedtuple('Segment', 'start end transition')
    if not isinstance(width, int):
        width = int(sr * width)
    width2 = width//2
    if not isinstance(transition, int):
        transition = int(sr * transition)
    num_channels, num_samples = X.shape
    if channels is None:
        channels = slice(0, num_channels)
    else:
        from . import data as D
        channels = [D.channels.index(c) for c in channels]

    slices = [
        Segment(start=max(0, n-width2), 
                end=min(num_samples, n+width2), 
                transition=transition)
        for n in range(0, num_samples, stride)
    ]
    repeats = len(slices)*batch_size
    surrogates = np.repeat(X, repeats).reshape(-1, repeats).transpose().reshape(
        len(slices), batch_size, num_channels, -1).swapaxes(1, 2)
    for i, segment in enumerate(slices):
        surrogates[i, channels] = np.array([
            partial_fft_surrogate_batch(x, batch_size, *segment)
            for x in X[channels]
        ])
    
    return surrogates


def distance(x, y, var=1.0):
    return np.mean((x-y)**2)/var

def iaaft(x, epsilon=0.0001, max_iter=50, verbose=0):
    import scipy.fftpack as sft
    d, n = np.inf, 0
    Amplitude = np.sort(x).astype(np.float64)
    Q84, Q50 = int(0.84*Amplitude.size), int(0.5*Amplitude.size)
    var = (Amplitude[Q84]-Amplitude[Q50])**2
    Fourier = sft.fft(x.astype(np.float64))
    Fourier_amplitudes = np.abs(Fourier)
    surrogate = fft_surrogate(f=Fourier)
    idx = np.argsort(surrogate)
    while epsilon < d and n < max_iter:
        surrogate[idx] = Amplitude
        f = sft.fft(surrogate.astype(np.float64))
        f = Fourier_amplitudes*np.exp(1.0j*np.angle(f))
        surrogate = sft.ifft(f).astype(np.float64)
        idx = np.argsort(surrogate)
        d = distance(Amplitude, surrogate[idx], var=var)
        n += 1
        if verbose:
            print('Iteration {n}, d = {d:.4g}'.format(n=n, d=d))
    return surrogate
