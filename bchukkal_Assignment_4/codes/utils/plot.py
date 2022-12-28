import os
import sys
cur_dir = os.path.dirname(os.path.realpath(__file__))
if cur_dir not in sys.path:
    sys.path.append(cur_dir)

import pandas as pd
import numpy as np
import math
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plt_plot(xs, ys, plot_func, colors, markers, labels, 
            title = None, yscale = None, ylb = None, yub = None,
            xlb = None, xub = None, xlabel = None, ylabel = None,
            legends = None, legendx = None, legendy = None, marker_sizes = np.ones(5)):

    # Plot a subplot graph          
    for x, y, c, m, label, s in zip(xs, ys, colors, markers, labels, marker_sizes):
        if plot_func == 'plot':
            plt.plot(x, y, color=c, marker=m, label=label)
        elif plot_func == 'scatter':
            plt.scatter(x, y, color=c, marker=m, s=s, label=label)

    if yscale:
        plt.yscale(yscale)
    if title:
        plt.title(title)
    if ylb or yub:
        plt.ylim(bottom=ylb, top=yub)        
    if xlb or xub:
        plt.xlim(left=xlb, right=xub)        
    
    if ylabel:
        plt.ylabel(ylabel, fontsize=11)
    if xlabel:
        plt.xlabel(xlabel, fontsize=11)

    if legends:
        #legend_x = legendx if legendx is not None else 2.0
        #legend_y = legendy if legendy is not None else 0.5
        if legendx and legendy:
            legend_x = legendx
            legend_y = legendy
            plt.legend(legends, loc='center right', bbox_to_anchor=(legend_x, legend_y))
        else:
            plt.legend()

    plt.axis('tight')        
    return    

def get_grid(x1_min, x1_max, x2_min, x2_max, step=0.02):
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, step),
                           np.arange(x2_min, x2_max, step))
    return xx1, xx2

def plot_decision_boundaries(xx1, xx2, num_cats, classifier, transformer = None, alpha = 0.4):
    colors = ('orange', 'indigo', 'purple', 'red')
    cmap = ListedColormap(colors[:num_cats])

    Xgrid = np.array([xx1.ravel(), xx2.ravel()]).T
    if transformer:
        Xgrid = transformer(Xgrid)
    y = classifier.predict(Xgrid)
    y = y.reshape(xx1.shape)
    plt.contourf(xx1, xx2, y, alpha=alpha, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())


