
# INSTALL
---------

## Using pip

Please use pip to install all necessary requirements to run this demo using
the command

  pip install -r requirements.txt

# ./download.py
---------------

This script lets you download all assets from a Dropbox repository as
`"blob.zip"` (5GB).  It is then automatically unzipped.


# ./signal-plotter
------------------

This is a bokeh application allowing you to view tfrecord files.  Start the
bokeh server with the command
  
  bokeh serve --show signal-plotter

The four panels show 30-second signal segments for EEG1, EEG2, EOG, and EMG.
Below, there's a slider that selects consecutive segments in the tfrecords
file.

# ./notebooks
-------------

We provide four notebooks illustrating parts of our analysis.

- Age and stage stats in the dataset.ipynb: This notebook analyzes the meta
  data in our dataset.  We check out how sleep-stage epochs are distributed
  among ages and classes.

- Convert checkpoint to keras model.ipynb:  This notebook uses our library
  functions to load a tensorflow checkpoint and convert it to a keras model
  blob.

- Examples of partial-surrogate analysis.ipynb:  In this notebook we illustrate
  how to load examples in our dataset.  We counterpose with an example the two
  methods of surrogate-based data augmentation, namely zeroing-out and
  ft-surrogates.

- Visualize layers as linear filters.ipynb:  In this notebook, we load a model
  and analyze the convolutional filters in the first layer.  For signals such
  convolutions can be interpreted as linear filters.  Using this theory, we
  display the frequency response and find signatures of sensitivity to
  physiological frequencies.  Note that this has not been reported in the
  paper.

# ./run.py
----------

While we performed our training using the google cloud infrastructure, we
provide the compound script `./run.py` that will let you do all computations
locally.  To start training your own model, you can simply execute

  ./python run.py train

There are plenty of command line arguments that will let you modify the
training.  Notably, you can provide a list of tfrecords as argument to
`--input` to train on the full dataset.  We designed such lists in shell
scripts to train our several splits.
