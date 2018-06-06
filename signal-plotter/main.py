
from sys import path
path.insert(0, '.')
from trainer import data as D
from trainer import model as M
from trainer import tools as tl
import numpy as np
import keras
from os.path import join, dirname
from bokeh.layouts import row, column, gridplot, Spacer
from bokeh.models import ColumnDataSource, Slider, CustomJS, HoverTool, Div
from bokeh.models.widgets import Button, TextInput
from bokeh.plotting import curdoc, figure

header = Div(text=open(join(dirname(__file__), "header.html")).read(), width=800)

def ssum(L):
    ret = l[0]
    for l in L[1:]:
        ret += l
    return ret

D.sr = 32.0

filename = './datasets/tfrecords/narco4.tfrecords'
record = None
source = None
surrogate_source = None
current_idx = 0
proba = None
surrogate_stride = 10
follow_interval = D.dt
signal_rollover = int(follow_interval*D.sr)
surrogate_rollover = int(follow_interval*D.sr/surrogate_stride)

default_time_surrogate = np.zeros(surrogate_rollover, dtype=np.float)
default_surrogate = np.zeros(surrogate_rollover, dtype=np.float)

modelfile = "./logs/cross_val/augment_0.4/split_0/ckpt-7000.h5"
model = keras.models.load_model(modelfile, custom_objects={'Scale': M.Scale})

def compute_surrogate_probabilities(*args, **kwargs):
    global record, current_idx
    Xin = np.array([source.data[c] for c in D.channels])
    surrogate_batches = tl.generate_partial_surrogate_batches(
        Xin, batch_size=32, stride=surrogate_stride, width=5.0,
        channels=['EEG1', 'EEG2'], transition=1.0, sr=D.sr)
    P = 100.0*np.array([
        model.predict(list(batch)).mean(axis=0)
        for batch in surrogate_batches
    ])
    for l, label in enumerate(D.events):
        record[label][current_idx] = P[:, l]
    record['time_surrogate'][current_idx] = D.dt*current_idx + np.arange(P.shape[0])/D.sr*surrogate_stride
    update(current_idx)

def filename_handler(attr, old, new):
    global record, proba
    record = D.read_tfrecords(new)
    example_slider.end = len(record['EEG1'])-1
    features = {
        tp: record[tp] for tp in D.channels
    }
    num_epochs = features['EEG1'].shape[0]
    print("Predicting probabilities in %i epochs"%num_epochs)
    record['proba'] = 100.0* model.predict(features)
    for label in D.events:
        record[label] = dict()
    record['time_surrogate'] = dict()

def set_view_position(example_idx):
    global surrogate_source, source, record, current_idx
    current_idx = example_idx
    cols = {channel: record[channel][example_idx] for channel in D.channels}
    cols['time'] = D.dt*example_idx + np.arange(cols['EEG1'].size)/D.sr
    scols = {'time_surrogate': record['time_surrogate'].get(example_idx, default_time_surrogate)}
    for label in D.events:
        scols[label] = record[label].get(example_idx, default_surrogate)
    txt = 'Prob ('+', '.join(['%.2f'%p for p in record['proba'][example_idx]])+'), '
    txt += D._decoder[record['target'][example_idx]]
    example_slider.title = txt
    if source is None:
        source = ColumnDataSource(cols)
    if surrogate_source is None:
        surrogate_source = ColumnDataSource(scols)
    return cols, scols

def update(t):
    cols, scols = set_view_position(t)
    source.stream(cols, signal_rollover)
    surrogate_source.stream(scols, surrogate_rollover)

filename_field = TextInput(value=filename, title="File name")
filename_field.on_change('value', filename_handler)

surrogate_button = Button(label="Compute Surrogate", button_type="success")
surrogate_button.on_click(compute_surrogate_probabilities)
select_file_button = Button(label="Select file", button_type="success")
select_file_button.callback = CustomJS(args={'filename_field': filename_field}, code = """
set_selection = function () {
    var fname = './datasets/tfrecords/'+fileSelector.files[0].name;
    filename_field.value = fname;
}
fileSelector = document.createElement('input');
fileSelector.setAttribute('type', 'file');
fileSelector.setAttribute('onchange', 'set_selection()');
fileSelector.click();
""")

example_slider = Slider(start=0, end=1, value=0, step=1, title="", show_value=False)

filename_handler(None, None, filename)
set_view_position(current_idx)

# build plots
plots = []
x_range = None
for c, channel in enumerate(D.channels):
    p = figure(sizing_mode='scale_width', plot_height=100, tools="", x_range=x_range, y_axis_location="left")
    x_range = p.x_range
    p.line(x='time', y=channel, alpha=1.0, line_width=1, color='black', source=source)
    p.yaxis.axis_label = channel+" (uV)"
    plots.append([p])

p = figure(sizing_mode='scale_width', plot_height=170, tools="", y_range=[10, 100], x_range=x_range, y_axis_location="left", y_axis_type='log')
x_range = p.x_range
for label in D.events:
    p.line(x='time_surrogate', y=label, alpha=1.0, line_width=3, color=D.color[label], source=surrogate_source, legend=label.lower())
p.yaxis.axis_label = 'Probability (%)'
plots.append([p])
p.add_tools(HoverTool(
    tooltips=[(label, "@"+label) for label in D.events]))
p.xaxis.axis_label = 'Time (sec.)'
p.legend.__setattr__('padding', 5)
p.legend.__setattr__('spacing', 0)
p.legend.__setattr__('label_height', 0)
p.legend.__setattr__('glyph_height', 10)
p.legend.__setattr__('background_fill_alpha', 0.5)
p.legend.__setattr__('label_text_alpha', 0.5)
x_range.follow = "end"
x_range.follow_interval = follow_interval
x_range.range_padding = 0

curdoc().add_root(
    column(
        header,
        gridplot(plots, toolbar_location="left", sizing_mode='scale_width'),
        row(surrogate_button, example_slider, filename_field, select_file_button, sizing_mode='scale_width'),
        sizing_mode='scale_width'
    )
)

controls = [example_slider]
for control in controls:
    control.on_change('value', lambda attr, old, new: update(new))
curdoc().title = filename
