#! /usr/bin/env python2

from __future__ import print_function

import os
import argparse
import numpy as np
# import logging
# logging.basicConfig(level=logging.INFO)

# logger = logging.getLogger(name='task')
from collections import defaultdict

import tensorflow as tf
from tensorflow.contrib.training.python.training import hparam
from tensorflow.contrib.learn.python.learn import learn_runner
from tensorflow.contrib.learn.python.learn.estimators import run_config
from tensorflow.contrib.learn.python.learn.utils import (
        saved_model_export_utils)

from . import data as D
from . import model as M
from . import tools as tl


def generate_experiment_fn(**experiment_args):

  def _experiment_fn(run_config, hparams):
    common_kw = {
        'batch_size': hparams.train_batch_size,
        'channel_subset': hparams.channel_subset,
    }
    train_input = D.generate_input_fn(hparams.train_data, train=True,
            balance_factor=hparams.balance_factor, augment=hparams.augment,  **common_kw)
    eval_input  = D.generate_input_fn(hparams.test_data, train=False, **common_kw)

    estimator   = M.build_estimator(
        config        = run_config,
        **hparams.__dict__
    )
    return tf.contrib.learn.Experiment(
        estimator,
        train_input_fn = train_input,
        eval_input_fn  = eval_input,
        **experiment_args
    )

  return _experiment_fn

# Build default_model_params from model functions
#   `M.Test_pipe`,
#   `M.end_pipe_layers`.
default_model_params = {}
for fn in [M.Test_pipe, M.end_pipe_layers]:
    default_model_params.update(tl.get_default_kwargs(fn))

default_model_params_by_type = defaultdict(dict)
for pname, pval in default_model_params.items():
    default_model_params_by_type[type(pval)][pname] = pval
default_model_params_by_type[float]['learning_rate'] = 0.05
# Construct inverse dictionary
type_of_model_params = {}
for typ, params_of_typ in default_model_params_by_type.items():
    for pname in params_of_typ:
        type_of_model_params[pname] = typ

def add_model_params_to(parser):
  # Model / Optimization
  model_params = {}
  for typ, params_of_typ in default_model_params_by_type.items():
      for pname, default in params_of_typ.items():
          pname_cli = pname.replace('_', '-')
          parser.add_argument('--'+pname_cli, type=typ, default=default)
          model_params[pname] = default

  parser.add_argument('--channel-subset', type=str, default=None)
  model_params['channel_subset'] = None
  parser.add_argument('--opt', choices=tl.optimizer.keys(), default='adagrad')
  model_params['opt'] = 'adagrad'
  return model_params


def parse_channel_subset(args):
    """Parse selected subset of channels.. for now just one!"""
    subset = args.channel_subset
    if subset is None:
        return None
    subset = [ch.strip() for ch in subset.split(',')]
    assert len(subset) == 1, 'Test only one channel [{}]'.format(subset)
    for ch in subset:
        assert ch in D.channels, "Illegal channel in args [%s]" % ch
    return subset


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  # ARGUEMENTS
  # In-/Output
  parser.add_argument('--train-data', type=str, nargs='+', required=True, help='GCS or local paths to training data')
  parser.add_argument('--num-epochs', type=int, help="""Feed to input_fn to get input only for num of epochs.""")
  parser.add_argument('--train-batch-size', type=int, default=40, help='Batch size for training steps')
  parser.add_argument('--job-dir', required=True, help='GCS location to write checkpoints and export models')
  parser.add_argument('--model-dir', default=None)

  parser.add_argument('--test-data', type=str, nargs='+', required=True, help='GCS or local paths to validation data')
  parser.add_argument('--balance-factor', type=float, default=1.0, help='')
  parser.add_argument('--augment', type=np.float, default=0.0, help='Probability to augment a signal repetition as surrogate.')

  # Logging
  parser.add_argument('--verbosity', choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'], default='INFO')

  # Experiment
  parser.add_argument('--train-steps-per-iteration', default=10, type=int, help='Frequency of running eval in steps.')
  parser.add_argument('--train-steps', type=int, default=None, help="""\
  Steps to run the training job for. If --num-epochs is not specified,
  this must be. Otherwise the training job will run indefinitely.\
  """)
  parser.add_argument('--eval-steps', type=int, default=1024)
  parser.add_argument('--export-format', choices=['JSON', 'CSV', 'EXAMPLE'], default='JSON', help='The input format of the exported SavedModel binary')
  parser.add_argument('--summary', type=str, nargs='+', default=['weights', 'gradients'])
  parser.add_argument('--dryrun', action='store_true')
  parser.add_argument('--tryrun', action='store_true')

  # Model / optimization
  add_model_params_to(parser)

  args = parser.parse_args()

  args.channel_subset = parse_channel_subset(args)

  args.model_dir = args.job_dir

  cf = tl.TfConfig()

  # Set python level verbosity
  tf.logging.set_verbosity(args.verbosity)
  #     Set C++ Graph Execution level verbosity
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(
      tf.logging.__dict__[args.verbosity] / 10)

  # SETUP EXPERIMENT AND RUN
  export_strategies = [
      saved_model_export_utils.make_export_strategy(
          D.SERVING_FUNCTIONS['TF_RECORD'](args.channel_subset),
          exports_to_keep = 1,
          default_output_alternative_key = None,
      )
  ]

  gen_exp_fn = generate_experiment_fn(
      train_steps_per_iteration = args.train_steps_per_iteration,
      train_steps        = args.train_steps,
      eval_steps         = args.eval_steps,
      export_strategies  = export_strategies
  )

  def run():
      learn_runner.run(
          gen_exp_fn,
          schedule   = 'continuous_train_and_eval',
          run_config = run_config.RunConfig(model_dir=args.job_dir),
          hparams    = hparam.HParams(**args.__dict__)
      )

  if args.tryrun:
      try:
          run()
      except Exception as e:
          print(e)
  else:
      run()
