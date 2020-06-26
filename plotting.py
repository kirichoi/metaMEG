# -*- coding: utf-8 -*-
"""
CCR plotting module

Kiri Choi (c) 2018
"""

import os, sys
import tellurium as te
import roadrunner
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


def plotAllProgress(listOfDistances, labels=None, SAVE_PATH=None):
    """
    Plots multiple convergence progress 
    
    :param listOfDistances: 2D array of distances
    :param labels: list of strings to use as labels
    :param SAVE_PATH: path to save plot
    """
    
    fig = plt.figure(figsize=(16, 6))
    for i in range(len(listOfDistances)):
        plt.plot(listOfDistances[i])
    if labels:
        plt.legend(labels)
    plt.xlabel("Generations", fontsize=15)
    plt.ylabel("Distance", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if SAVE_PATH is not None:
        if os.path.splitext(SAVE_PATH)[1] == '':
            plt.savefig(os.path.join(SAVE_PATH, 'images/convergence.pdf'), bbox_inches='tight')
        else:
            plt.savefig(SAVE_PATH, bbox_inches='tight')
    plt.show()
    

def plotProgress(distance, SAVE_PATH=None):
    """
    Plots convergence progress
    
    :param distance: array of distances
    :param model_type: reference model type, e.g. 'FFL', 'Linear', etc.
    :param SAVE_PATH: path to save plot
    """
    
    fig = plt.figure(figsize=(16, 6))
    plt.plot(distance)
    plt.xlabel("Generations", fontsize=15)
    plt.ylabel("Distance", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if SAVE_PATH is not None:
        if os.path.splitext(SAVE_PATH)[1] == '':
            plt.savefig(os.path.join(SAVE_PATH, 'images/convergence.pdf'), bbox_inches='tight')
        else:
            plt.savefig(SAVE_PATH, bbox_inches='tight')
    plt.show()
    

def plotResidual(realModel, ens_model, ens_dist, SAVE_PATH=None):
    """
    Plots residuals
    
    :param realModel: reference model
    :param ens_model: model ensemble
    :param ens_dist: model distances
    :param model_typ: reference model type
    :param SAVE_PATH: path to save plot
    """
    
    try:
        r_real = te.loada(realModel)
    except:
        r_real = te.loads(realModel)
    result_real = r_real.simulate(0, 100, 100)
    
    top_result = []
    top_diff = []
    
    for i in range(len(ens_model)):
        r = te.loada(ens_model[np.argsort(ens_dist)[i]])
        top_sim = r.simulate(0, 100, 100)
        top_result.append(top_sim)
        top_diff.append(np.subtract(result_real[:,1:], top_sim[:,1:]))

    percentage = 0.1#float(pass_size)/ens_size
    
    ave_diff = np.average(top_diff[:int(len(ens_model)*percentage)], axis=0)
    
    fig = plt.figure(figsize=(10, 6))
    plt.plot(ave_diff)
    plt.xlabel("Time (s)", fontsize=15)
    plt.ylabel("Residual", fontsize=15)
    plt.legend(r.getFloatingSpeciesIds())
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if SAVE_PATH is not None:
        if os.path.splitext(SAVE_PATH)[1] == '':
            plt.savefig(os.path.join(SAVE_PATH, 'images/average_residual.pdf'), bbox_inches='tight')
        else:
            plt.savefig(SAVE_PATH, bbox_inches='tight')
    plt.show()
    
    
def plotDistanceHistogram(ens_dist, nbin=25, SAVE_PATH=None):
    """
    """
    
    fig = plt.figure(figsize=(10, 6))
    plt.hist(ens_dist, bins=nbin, density=True)
    plt.xlabel("Distance", fontsize=15)
    plt.ylabel("Normalized Frequency", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if SAVE_PATH is not None:
        if os.path.splitext(SAVE_PATH)[1] == '':
            plt.savefig(os.path.join(SAVE_PATH, 'images/distance_hist.pdf'), bbox_inches='tight')
        else:
            plt.savefig(SAVE_PATH, bbox_inches='tight')
    plt.show()


def plotDistanceHistogramWithKDE(dist_top, minInd, nbin=40, SAVE_PATH=None):
    """
    Plot histogram of collected model distances with cutoff using kernel density
    estimation to form an ensemble visualized
    
    :param dist_top: list of distances
    :param minInd: list of extrema
    :param nbin: number of bins
    :param SAVE_PATH: path to save plot
    
    """
    
    fig = plt.figure(figsize=(10, 6))
    hist = plt.hist(dist_top, bins=nbin, density=True)
    plt.vlines(dist_top[minInd[0]], 0, np.max(hist[0]), linestyles='dashed')
    plt.xlabel("Distance", fontsize=15)
    plt.ylabel("Normalized Frequency", fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if SAVE_PATH is not None:
        if os.path.splitext(SAVE_PATH)[1] == '':
            plt.savefig(os.path.join(SAVE_PATH, 'images/distance_hist_w_KDE.pdf'), bbox_inches='tight')
        else:
            plt.savefig(SAVE_PATH, bbox_inches='tight')
    plt.show()


def plotNetwork(path, scale=1.5):
    """
    Plot a network diagram
    
    :param path: path to a model
    :param scale: diagram scale
    """
    
    import netplotlib as npl
    
    net = npl.Network(path)
    net.scale = scale
    net.draw()


def plotNetworkEnsemble(path, index=None, threshold=0., scale=1.5):
    """
    Plot network ensemble diagram
    
    :param path: path to output folder
    :param index: index of models to be included in the diagram
    :param threshold: threshold of reactions to be plotted
    :param scale: diagram scale
    """
    
    import netplotlib as npl
    
    model_dir = os.path.join(path, 'models')
    
    modelfiles = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
    
    modelcontent = []
    for i in modelfiles:
        sbmlstr = open(os.path.join(model_dir, i), 'r')
        modelcontent.append(sbmlstr.read())
        sbmlstr.close()
    
    if index >= len(modelcontent):
        raise Exception("Specified index value is larger than the size of the list")
        
    net = npl.NetworkEnsemble(modelcontent[:index])
    net.plottingThreshold = threshold
    net.scale = scale
    net.drawWeightedDiagram()


def plotNetworkComparison(path1, path2, title=None, scale=1., SAVE_PATH=None):
    """
    Plot real network diagram and best network diagram side-by-side
    
    :param path1: path or Antimony string of a model
    :param path2: path or Antimony string of a model
    :param title: list of custom titles for each diagram
    :param scale: diagram scale
    :param SAVE_PATH: path to save diagram
    """
    
    import netplotlib as npl
    
    if title != None:
        assert(len(title) == 2)
    else:
        title = ['Original', 'Best Output']
    
    fig, ax = plt.subplots(ncols=2, figsize=(22,12))
    ax[0].axis('off')
    ax[0].set_title(title[0], fontsize=15)
    net1 = npl.Network(path1)
    net1.customAxis = ax[0]
    net1.scale = scale
    pos1 = net1.getLayout()
    net1.draw()
    
    ax[1].axis('off')
    ax[1].set_title(title[1], fontsize=15)
    net2 = npl.Network(path2)
    net2.customAxis = ax[1]
    net2.scale = scale
    net2.setLayout(pos1)
    net2.draw()
    
    if SAVE_PATH is not None:
        if os.path.splitext(SAVE_PATH)[1] == '':
            plt.savefig(os.path.join(SAVE_PATH, 'images/net_comp_real_best.pdf'), bbox_inches='tight')
        else:
            plt.savefig(SAVE_PATH, bbox_inches='tight')
    
    
def plotWeightedNetwork(path1, path2, title=None, scale=1., threshold=0.25, SAVE_PATH=None):
    """
    Plot real network diagram and ensemble weighted network diagram side-by-side
    
    :param path1: path or Antimony string of a model
    :param path2: list of paths or Antimony strings of models
    :param title: list of custom titles for each diagram
    :param scale: diagram scale
    :param threshold: plotting threshold for edges
    :param SAVE_PATH: path to save diagram
    """
    
    import netplotlib as npl
    
    if title != None:
        assert(len(title) == 2)
    else:
        title = ['Original', 'Ensemble']
    
    fig, ax = plt.subplots(ncols=2, figsize=(22,12))
    ax[0].axis('off')
    ax[0].set_title(title[0], fontsize=15)
    net1 = npl.Network(path1)
    net1.customAxis = ax[0]
    net1.scale = scale
    net1.tightLayout = True
    pos1 = net1.getLayout()
    net1.draw()
    
    ax[1].axis('off')
    ax[1].set_title(title[1], fontsize=15)
    net2 = npl.NetworkEnsemble(path2)
    net2.customAxis = ax[1]
    net2.edgelw = 1
    net2.scale = scale
    net2.plottingThreshold = threshold
    net2.tightLayout = True
    net2.disableReactionEdgeLabel=True
    net2.setLayout(pos1)
    net2.drawWeightedDiagram()
    
    if SAVE_PATH is not None:
        if os.path.splitext(SAVE_PATH)[1] == '':
            plt.savefig(os.path.join(SAVE_PATH, 'images/net_comp_real_ensemble.pdf'), bbox_inches='tight')
        else:
            plt.savefig(SAVE_PATH, bbox_inches='tight')
    
