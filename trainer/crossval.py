
import os
from sklearn import metrics
from . import data as D
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from . import plotting

SURROGATES = 'surrogates iaaft'.split() + [None]
TESTSET_VALS = (False, True)
LABELS = 'Wake S1 S2 S3 S4 REM'.split()
REDUCED_LABELS = 'Wake Light Deep REM'.split()

_REDUCTION = dict([
    ('Wake', 'Wake'),
    ('S1', 'Light'),
    ('S2', 'Light'),
    ('S3', 'Deep'),
    ('S4', 'Deep'),
    ('REM', 'REM')
])
_REDUCTION = {
    LABELS.index(full): REDUCED_LABELS.index(reduced)
    for full, reduced in _REDUCTION.items()
}
reduce_stages = np.vectorize(_REDUCTION.get)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plotting.colorscheme['accuracy']):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, None]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, '{}%'.format(int(100.0*cm[i, j])),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize=12)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return

def plot_mats(df, title_prefix=None):
    title = 'Training set'
    if title_prefix is not None:
        title = title_prefix+', '+title
    DF = df[df.testset == False]
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    cm = metrics.confusion_matrix(DF.truth, DF.prediction)
    plot_confusion_matrix(cm, labels, normalize=True, title=title)
    plt.clim(0, 1)

    DF = df[df.testset == True]
    plt.subplot(122)
    cm = metrics.confusion_matrix(DF.truth, DF.prediction)
    plot_confusion_matrix(cm, labels, normalize=True, title='Test set')
    plt.clim(0, 1)
    plt.tight_layout()
    return

def plot_mats_reduced(df, title_prefix=None):
    title = 'Training set'
    if title_prefix is not None:
        title = title_prefix+', '+title
    DF = df[df.testset == False]
    plt.figure(figsize=(15, 6))
    plt.subplot(121)
    cm = metrics.confusion_matrix(reduce_stages(DF.truth), reduce_stages(DF.prediction))
    plot_confusion_matrix(cm, reduced_labels, normalize=True, title=title)
    plt.clim(0, 1)

    DF = df[df.testset == True]
    plt.subplot(122)
    cm = metrics.confusion_matrix(reduce_stages(DF.truth), reduce_stages(DF.prediction))
    plot_confusion_matrix(cm, reduced_labels, normalize=True, title='Test set')
    plt.clim(0, 1)
    plt.tight_layout()
    return
    

def f1_score(df, average='macro'):
    return metrics.f1_score(df.truth, df.prediction, average=average)


class _Crossval:

    def __init__(self, data=None):
        self._data = data
        self._accuracy = None
        self._reduced_accuracy = None
        self._total_accuracy = None

    @property
    def data(self):
        return self._data

    @property
    def accuracy(self):
        if self._accuracy is None:
            df = self.data
            assert sum(df.pid=='brux1') == 0
            mindex = pd.MultiIndex.from_product([D.age_group_bins, TESTSET_VALS, LABELS, LABELS])
            mindex.names = ['age_group', 'testset', 'groundtruth', 'prediction']
            acc = pd.Series(data=None, index=mindex)
            for Bin in D.age_group_bins:
                for testset in TESTSET_VALS:
                    mask = (df.testset == testset) & (df.age_group == Bin)
                    dfm = df[mask]
                    cm = metrics.confusion_matrix(dfm.truth, dfm.prediction)
                    cm = cm.astype(np.float) / cm.sum(axis=1)[:, None]
                    for true_label, cm_l in zip(LABELS, cm):
                        for pred_label, cm_lp in zip(LABELS, cm_l):
                            acc[(Bin, testset, true_label, pred_label)] = cm_lp
            self._accuracy = acc
        return self._accuracy

    @property
    def reduced_accuracy(self):
        if self._reduced_accuracy is None:
            df = self.data
            assert sum(df.pid=='brux1') == 0
            mindex = pd.MultiIndex.from_product(
                [D.age_group_bins, TESTSET_VALS, REDUCED_LABELS, REDUCED_LABELS])
            mindex.names = ['age_group', 'testset', 'groundtruth', 'prediction']
            acc = pd.Series(data=None, index=mindex)
            for Bin in D.age_group_bins:
                for testset in TESTSET_VALS:
                    mask = (df.testset == testset) & (df.age_group == Bin)
                    dfm = df[mask]
                    cm = metrics.confusion_matrix(reduce_stages(dfm.truth), reduce_stages(dfm.prediction))
                    cm = cm.astype(np.float) / cm.sum(axis=1)[:, None]
                    for true_label, cm_l in zip(REDUCED_LABELS, cm):
                        for pred_label, cm_lp in zip(REDUCED_LABELS, cm_l):
                            acc[(Bin, testset, true_label, pred_label)] = cm_lp
            self._reduced_accuracy = acc
        return self._reduced_accuracy

    @property
    def total_accuracy(self):
        """returns accuracy not stratified by age bins"""
        if self._total_accuracy is None:
            df = self.data
            assert sum(df.pid=='brux1') == 0
            mindex = pd.MultiIndex.from_product(
                [TESTSET_VALS, LABELS, LABELS])
            mindex.names = ['testset', 'groundtruth', 'prediction']
            acc = pd.Series(data=None, index=mindex)
            for testset in TESTSET_VALS:
                dfm = df[df.testset == testset]
                cm = metrics.confusion_matrix(dfm.truth, dfm.prediction)
                cm = cm.astype(np.float) / cm.sum(axis=1)[:, None]
                for true_label, cm_l in zip(LABELS, cm):
                    for pred_label, cm_lp in zip(LABELS, cm_l):
                        acc[(testset, true_label, pred_label)] = cm_lp
            self._total_accuracy = acc
        return self._total_accuracy

    @staticmethod
    def plot_accuracy(cm, classes, cmap=plotting.colorscheme['accuracy'],
            normalize=False, colorbar=True, fontsize=12, vmin=None, vmax=None):
        tick_marks = np.arange(len(classes))
        if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, None]
        ax = plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=vmin,
                vmax=vmax)
        if colorbar: plt.colorbar()
        plt.xticks(tick_marks, classes, rotation=30, fontsize=fontsize)
        plt.yticks(tick_marks, classes, fontsize=fontsize)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, '{}%'.format(int(100.0*cm[i, j])),
                     ha="center", va='center', fontsize=fontsize,
                     color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True class', fontsize=fontsize)
        plt.xlabel('Predicted class', fontsize=fontsize)
        return ax


class Crossval(_Crossval):

    def __init__(self, job_name, sampling_rate=128.0, surrogates=None):
        super().__init__()
        assert surrogates in SURROGATES, 'unknown %s'%surrogates
        self.surrogates = surrogates
        self.job_name = job_name
        self.sampling_rate = sampling_rate
        assert os.path.exists(self.evaluation_dir), self.evaluation_dir

    @property
    def evaluation_dir(self):
        if self.sampling_rate == 32.0:
            d = "./logs/sr32/Joined/cross_val/%s/" % self.job_name
        else:
            d = "./logs/Joined/cross_val/%s/" % self.job_name
        assert os.path.exists(d), d
        return d

    @property
    def evaluation_file(self):
        if self.surrogates is not None:
            basename = 'eval-%s.pkl'%self.surrogates
        else:
            basename = 'eval.pkl'
        return os.path.join(self.evaluation_dir, basename)

    @property
    def data(self):
        if self._data is None:
            self._data = pd.read_pickle(self.evaluation_file)
            meta = pd.read_csv('./Datasets/patient_info.csv')
            meta.pid = meta.pid.apply(lambda x: x.lower())
            self._data = pd.merge(self._data, meta, on='pid')
        return self._data

    def figure_file(self, basename):
        d = os.path.join(self.evaluation_dir, 'Fig')
        if not os.path.exists(d):
            os.makedirs(d)
        return os.path.join(d, basename)
