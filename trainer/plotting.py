
import matplotlib.pyplot as plt

stage_color = {
    'Wake': '#EEBD6B',
    'S1': '#BFE6EF',
    'S2': '#447695',
    'S3': '#04082E',
    'S4': '#04082E',
    'REM': '#359D93',
}

stage_line_type = {
    'Wake': 'o',
    'S1': 'o',
    'S2': '^',
    'S3': '>',
    'S4': 'v',
    'REM': 's',
}

set_color = {
    'train': 'b',
    'eval': 'r'
}
set_color[True] = set_color['eval']
set_color[False] = set_color['train']

set_marker = {
    'train': '^',
    'eval': '>'
}
set_marker[True] = set_marker['eval']
set_marker[False] = set_marker['train']

colorscheme = {
    'frequency': plt.cm.Blues,
    'accuracy': plt.cm.Oranges,
    'relative': plt.cm.Greens
}
