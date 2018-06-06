#!/usr/bin/env python

from __future__ import print_function

import os

import trainer.data as D 
import trainer.task as T 
import trainer.model as M 
import trainer.tools as tl

import argparse
from tensorflow.contrib.learn.python.learn.estimators import run_config
import logging
logging.basicConfig(level=logging.DEBUG)

# Request user arguments
parser = argparse.ArgumentParser()
parser.add_argument('mode', choices=(
    'train',
    'eval',
    'predict',
    'hopt',
    'staircase',
    'evaluations',
    'download'
))
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--num-steps', type=int, default=10000)
parser.add_argument('--model-dir', type=str, default='./logs/test')
parser.add_argument('--job-name', type=str,  default=None)
parser.add_argument('--input', type=str, nargs='+', default=['./datasets/tfrecords/n2.tfrecords'])
parser.add_argument('--output', type=str, default=None)
parser.add_argument('--cutoff', type=float, default=0.0)
parser.add_argument('--balance-factor', type=float, default=1.0)
parser.add_argument('--summary', type=str, nargs='+', default=['weights', 'gradients'])

# arguments of model / optimization
T.add_model_params_to(parser)

# Parse arguments
args = parser.parse_args()
args.channel_subset = T.parse_channel_subset(args)

# RunConfig points to dir the model is found/saved
config = run_config.RunConfig(model_dir=args.model_dir)

# Select input generator from extension
print(args.input)
if not args.mode in ('staircase', 'evaluations'):
    input_fn = D.generate_input_fn(
        args.input,
        args.batch_size,
        train = (args.mode=='train'),
        balance_factor = args.balance_factor,
        channel_subset = args.channel_subset
    )


def train(net, input_fn, steps):
    net.train(input_fn=input_fn, steps=steps)


def predict(net, input_fn):
    return net.predict(input_fn=input_fn)


def evaluate(net, input_fn):
    return net.evaluate(input_fn=input_fn)


def hopt(net, input_fn, steps):
    pass

nn = M.build_estimator(config, **args.__dict__)

if 'train' == args.mode:
    train(nn, input_fn, args.num_steps)

elif 'eval' == args.mode:
    score = evaluate(nn, input_fn)
    print('score:', score)

elif 'predict' == args.mode:
    prediction = predict(nn, input_fn)
    for i, p in enumerate(prediction):
        print(p)
        if i > 15:
            break

elif 'hopt' == args.mode:
    hopt(nn, input_fn)

elif 'staircase' == args.mode:
    job = tl.HPJob.from_gcloud_job(args.job_name)
    job.plot_objective(filename=args.output, cutoff=args.cutoff)

elif 'evaluations' == args.mode:
    job = tl.HPJob.from_gcloud_job(args.job_name)
    job.plot_evaluations(filename=args.output)

elif 'download' == args.mode:
    assert args.input is not None and args.output is not None
    tl.gs.cp(args.input, args.output)

else:
    # >3.6
    # AttributeError(f'Mode {args.mode} not understood.')
    AttributeError('Mode {} not understood.'.format(args.mode))
