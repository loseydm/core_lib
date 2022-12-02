import os
import time
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from typing import List

def add_custom_legend(ax, colors, labels, linewidth=2, loc='upper left', fancybox=True, bbox_to_anchor=(1.0, 1.00), shadow=True,
                      fontsize=10):
    """
    Adds a custom legend, with specified colors and the corresponding labels, to the passed axis.    

    Parameters
    ----------
    ax : matplotlib axis 
    colors : colors used for plot
    labels : labels for each color
    linewidth : linewidth of color on the legend
    loc : tuple, optional
        coordinate to anchor legend. The default is 'upper left'.
    fancybox : TYPE, optional
        DESCRIPTION. The default is True.
    bbox_to_anchor : tuple, optional
        where to anchor legend. The default is (1.0, 1.00), which is the upper right portion
        of the plot. When loc is 'upper left', this defaults to the legend being placed
        outside of the plot on the upper left.
    shadow : bool, optional
        show legend box with a shadow? The default is True.
    fontsize : int, optional
        Legend fontsize. The default is 10.

    Returns
    -------
    None. Adds legend to axis

    """
    custom_lines = calc_handles_for_custom_legend(colors, labels=labels, linewidth=linewidth)
    if ax is None: plt.legend(handles=custom_lines, loc=loc, fancybox=fancybox, bbox_to_anchor=bbox_to_anchor, shadow=shadow, fontsize=fontsize)
    else:          ax.legend(handles=custom_lines, loc=loc, fancybox=fancybox, bbox_to_anchor=bbox_to_anchor, shadow=shadow,fontsize=fontsize)

def save_fig(title:str=None, tight_layout:bool=True, rect:List[int]=None,save_type:List[str]=None, 
             save_svg:bool=True, save_png:bool=False, save_folder:str=None):
    """
    Saves current matplotlib figure    

    Parameters
    ----------
    title : str, optional
        title of plot, will be saved under this name. The default is to save it as the current time.
    tight_layout : bool, optional
        Will call plt.tightlayout() to fix formatting. The default is True.
    rect : List[int], optional
        Lenghth of 4 ints denoting the rectangle to save. Defaults to a rectangle that doesn't
        crop the figure.
    save_type : List[str], optional
        . The default is None.
    save_svg : bool, optional
        Save fig as an svg file. The default is True.
    save_png : bool, optional
        Save fig as a png file. The default is False.
    save_folder : str, optional
        Where to save the file. The default is the current folder.
    """
    if (not save_png) and (not save_svg): save_png = True
    if save_folder is None: save_folder = './figs'
    os.makedirs(save_folder, exist_ok=True)
    if rect is None: rect = [0, 0, 1, 1]
    if title is None: title = time.time()
    if tight_layout: plt.tight_layout(rect=rect)
    save_type = []
    if save_png: save_type.append('png')
    if save_svg: save_type.append('svg')
    for st in save_type:
        save_path = os.path.join(save_folder, f'{title}.{st}')
        # bbox_inches='tight' crops the plot down based on the extents of the artists in the plot.
        # https://stackoverflow.com/questions/44642082/text-or-legend-cut-from-matplotlib-figure-on-savefig
        plt.gcf().savefig(save_path, dpi=plt.gcf().dpi, bbox_inches='tight')
        plt.ion()
    return save_path

###############################################
# Fonts, Colors, Plotting #####################
###############################################
def set_default_fonts():
    """ Sets some default parameters for matplotlib. """
    matplotlib.rc('font', family='sans-serif') 
    matplotlib.rc('font', serif='Arial') 
    matplotlib.rcParams.update({'font.size': 12})
    matplotlib.rcParams['svg.fonttype'] = 'none'
    matplotlib.rc('lines', solid_capstyle='butt')
    

def set_standard_fonts(ax=None, axis_label_fontsize=7, axis_label_fontnames='Arial',
                       tick_fontsize=5):
    """ Quick way to set the tick labels, axis labels fontsizes and fontnames."""
    axes = _to_ax_iterable(ax)
    for ax in axes:
        ax.xaxis.get_label().set_fontname('Arial')
        ax.yaxis.get_label().set_fontname('Arial')
        set_tick_label_fontnames(both='Arial', ax=None)
        ax.xaxis.get_label().set_fontsize(axis_label_fontsize)
        ax.yaxis.get_label().set_fontsize(axis_label_fontsize)
        for tick in ax.get_xticklabels(): tick.set_fontsize(tick_fontsize)
        for tick in ax.get_yticklabels(): tick.set_fontsize(tick_fontsize)
        
def set_tick_label_fontnames(both='Arial', ax=None):
    axes = _to_ax_iterable(ax)
    for ax in axes:
        for tick in ax.get_xticklabels():
            tick.set_fontname(both)
        for tick in ax.get_yticklabels():
            tick.set_fontname(both)

def hide_spine(tblr='tr', ax=None):
    """ 
    Hides the ax spine. tblr is 'top' 'bottom' 'left' 'right' list """
    for top_bttm_left_right in tblr:
        if   top_bttm_left_right.startswith('l'): ax.spines['left'].set_visible(False)
        elif top_bttm_left_right.startswith('r'): ax.spines['right'].set_visible(False)
        elif top_bttm_left_right.startswith('t'): ax.spines['top'].set_visible(False)
        elif top_bttm_left_right.startswith('b'): ax.spines['bottom'].set_visible(False)
        else: raise ValueError(tblr)

def _is_iterable(arr):
    try:
        iter(arr)
        return True
    except TypeError: return False

def _to_ax_iterable(axes):
    if axes is None: axes = plt.gca()
    if _is_iterable(axes): return axes
    else: return [axes]

def hide_tick_marks(xy, ax=None):
    """ hides tick marks of ax. 
        xy  = 'x', 'y', or 'xy' denoting which axis to hide tickmarks of
    """
    if ax is None: ax = plt.gca()
    axes = _to_ax_iterable(ax)
    for _, ax in enumerate(axes):
        if   xy == 'x': ax.tick_params(axis='x', which='both',length=0)
        elif xy == 'y': ax.tick_params(axis='y', which='both',length=0)
        elif (xy == 'xy') or (xy == 'both'): ax.tick_params(axis='both', which='both',length=0)

    
def set_tick_locations(locations, xy, ax=None):
    """ sets tick locations for ax """
    axes = _to_ax_iterable(ax)
    for _, ax in enumerate(axes):
        if 'x' in xy: ax.set_xticks(locations)
        if 'y' in xy: ax.set_yticks(locations)

def calc_zero_centered_hist_bins(y, bin_width):
    """ creates bins for a histogram that are centered at zero.
        y is data that needs to be bined. """
    bins = list(np.arange( (np.floor(np.min(y)) // bin_width) * bin_width , 0, bin_width)) + list(np.arange(0, (np.ceil(np.max(y)) // bin_width) * bin_width + 2 * bin_width, bin_width))
    assert np.min(y) > bins[0]
    assert np.max(y) < bins[-1], (np.max(y), bins[-1])
    return bins

# Quick conversions
def mm_to_inch(mm): return mm / 25.4
def cm_to_inch(cm): return cm / 2.54
def inch_to_cm(inch): return inch * 2.54


def calc_handles_for_custom_legend(colors, labels, linewidth=2):
    if type(colors) is str:
        colors = [colors]
        labels = [labels]
    assert len(colors) == len(labels)
    custom_lines = []
    for l, c in zip(labels, colors):
        custom_lines.append(matplotlib.lines.Line2D([0], [0], color=c, lw=linewidth, label=l))
    return custom_lines

def hex_to_rgb(hex_str):
    """ https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python """
    h = hex_str.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 256 for i in (0, 2, 4))

def hex_to_rgba(hex_str, alpha):
    """ https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python """
    h = hex_str.lstrip('#')
    return list((int(h[i:i+2], 16) / 256 for i in (0, 2, 4))) + [alpha]

###################################
## Plotting: Tick Marks & Labels ##
###################################
def hide_tick_marks_and_labels(xy='xy', ax=None):
    """ hides tick marks and labels by setting them to be an empty list"""
    if ax is None: ax = plt.gca()
    if xy == 'xy':
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    elif xy == 'x': ax.get_xaxis().set_ticks([])
    elif xy == 'y': ax.get_yaxis().set_ticks([])
    elif xy == 'z': ax.get_zaxis().set_ticks([])
    elif xy == 'xy' or xy == 'yx':
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
    elif xy == 'xyz':
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.get_zaxis().set_ticks([])
    else: raise ValueError(xy)

def hide_tick_marks(xy, ax=None):
    if ax is None: ax = plt.gca()
    axes = _to_ax_iterable(ax)
    for _, ax in enumerate(axes):
        if   xy == 'x': ax.tick_params(axis='x', which='both',length=0)
        elif xy == 'y': ax.tick_params(axis='y', which='both',length=0)
        elif (xy == 'xy') or (xy == 'both'): ax.tick_params(axis='both', which='both',length=0)

def hide_tick_labels(xy, axes=None, set_invisible=False):
    """ Same as hide_tick_marks but just sets them to be invisible instead of removing them.
        Use when sharing axes and want to show on only 1. """
    assert 'x' in xy or 'y' in xy, xy
    axes = _to_ax_iterable(axes)

    for ax in axes:
        if set_invisible:
            if 'x' in xy: plt.setp(ax.get_xticklabels(), visible=False)
            if 'y' in xy: plt.setp(ax.get_yticklabels(), visible=False)
            else: raise ValueError(xy)
        else:
            if   'x' in xy: ax.set_xticklabels([])
            elif 'y' in xy: ax.set_yticklabels([])
            else: raise ValueError(xy)

def show_tick_labels(xy='xy', ax=None):
    assert 'x' in xy or 'y' in xy, xy
    axes = _to_ax_iterable(ax)
    for ax in axes:
        if 'x' in xy: ax.xaxis.set_tick_params(labelbottom=True)
        if 'y' in xy: ax.yaxis.set_tick_params(labelbottom=True)

def hide_tick_labels_invis(xy, axes=None):
    """ https://stackoverflow.com/questions/4209467/matplotlib-share-x-axis-but-dont-show-x-axis-tick-labels-for-both-just-one
        Same as hide_tick_marks but just sets them to be invisible instead of removing them.
        Use when sharing axes, as this won't effect the other shared axis
    """
    assert 'x' in xy or 'y' in xy, xy
    if axes is None: axes = plt.gca()
    axes = _to_ax_iterable(axes)

def set_tick_label_fontsize(both=None, x_tick_label_fontsize=None, y_tick_label_fontsize=None, ax=None):
    """ Sets the size of the tick marks for each subplot """
    axes = _to_ax_iterable(ax)
    if both is not None:
        if type(both) is bool:
            raise ValueError("Both is supposed to be an int, not bool")
        x_tick_label_fontsize = both
        y_tick_label_fontsize = both
    try:   len(axes)
    except TypeError: raise
    for ax in axes:
        for tick in ax.xaxis.get_major_ticks():
            if x_tick_label_fontsize is not None: tick.label1.set_fontsize(x_tick_label_fontsize)
        for tick in ax.yaxis.get_major_ticks():
            if y_tick_label_fontsize is not None: tick.label1.set_fontsize(y_tick_label_fontsize)

def plot_bar_with_stars(mean, top_data_lim, pval, left_dist_mean=0, ax=None, color=None, star_fontsize=10, linewidth=2):
    """ plots line with corresponding p-value stars."""
    if ax is None: ax = plt.gca()
    if color is None: color = 'k'
    if pval > 0.05: star_fontsize = 5
    stars = pval_to_stars(pval)   
    add_text( (mean + left_dist_mean) / 2, top_data_lim, stars, ax=ax, fontsize=star_fontsize, coord_type='data', va='bottom', 
             ha='center', fontname='Arial', color=color)
    ax.plot( [left_dist_mean, mean], [top_data_lim, top_data_lim], color=color, linewidth=linewidth, solid_capstyle='butt')

def pval_to_stars(pval):
    if pval < 0.001: stars = '***'
    elif pval < 0.01: stars = '**'
    elif pval < 0.05: stars = '*'
    else: stars = 'n.s.'
    return stars

def add_text(x, y, text, ha='auto', va='auto', ax=None, fontsize=10, coord_type='axis', fontname='Arial', **kwargs):
    """ 
    Adds text to ax, but automated so the text is less likely to run over the edge by changing the vertical and horizontal
    alighment based on the passed x and y location.
    
        ha in 'left', 'center', right
        va in 'top', 'bottom """
    if ax is None: ax = plt.gca()
    if ha == 'auto':
        if x < 0.5: ha = 'left'
        elif x > 0.5: ha = 'right'
        elif x == 0.5: ha = 'center'
    if va == 'auto':
        if y < 0.5: va = 'bottom'
        elif y > 0.5: va = 'top'
        elif y == 0.5: va = 'center'

    if coord_type in {'axis', 'ax'}:
            ax.text(x, y, text, verticalalignment=va, horizontalalignment=ha, fontsize=fontsize, transform=ax.transAxes, fontname=fontname, **kwargs)
    elif coord_type == 'data':
            ax.text(x, y, text,
        verticalalignment=va, horizontalalignment=ha, fontsize=fontsize, fontname=fontname, **kwargs)
    else: raise ValueError(coord_type)

def add_lead_zeros(integer, num_places=4): 
    """ Adds leading zeros to an integer, returning a string """
    return f"%0{num_places}d" % (integer,)

def set_tick_labels(labels, xy, ax=None):
    """ sets tick label locations for the specified axis on the the passed ax """
    axes = _to_ax_iterable(ax)
    for _, ax in enumerate(axes):
        if 'x' in xy: ax.set_xticklabels(labels)
        if 'y' in xy: ax.set_yticklabels(labels)
