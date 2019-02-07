# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 23:56:16 2019

@author: kirichoi
"""

import os, sys
import tellurium as te
import roadrunner
import numpy as np
import antimony
import scipy.optimize
import networkGenerator as ng
import plotting as pt
import ioutils
import analysis
import matplotlib.pyplot as plt
import time
import copy


def f1(k_list, *args):
    global counts
    global countf
    
    args[0].reset()
    
    args[0].setValues(args[0].getGlobalParameterIds(), k_list)
    
    try:
        args[0].steadyState()
        objCC = args[0].getScaledConcentrationControlCoefficientMatrix()
        objCC[np.abs(objCC) < 1e-16] = 0 # Set small values to zero
        
        dist_obj = ((np.linalg.norm(realConcCC - objCC))/
                    (1 + np.sum(np.equal(np.sign(np.array(realConcCC)), np.sign(np.array(objCC))))))
                    
    except:
        countf += 1
        dist_obj = 10000
        
    counts += 1
    
    return dist_obj


def callbackF(X, convergence=0.):
    global counts
    global countf
    print(str(counts) + ", " + str(countf))
    return False


def initialize():
    global countf
    global counts
    
    numBadModels = 0
    numGoodModels = 0
    numIter = 0
    
    ens_dist = np.empty(ens_size)
    ens_model = np.empty(ens_size, dtype='object')
    ens_rl = np.empty(ens_size, dtype='object')
    rl_track = []
    
    # Initial Random generation
    while (numGoodModels < ens_size):
        rl = ng.generateReactionList(ns, nr, realBoundaryIdsInd)
        st = ng.getFullStoichiometryMatrix(rl, ns).tolist()
        stt = ng.removeBoundaryNodes(np.array(st))
        # Ensure no redundant model
        while (stt[1] != realFloatingIdsInd or stt[2] != realBoundaryIdsInd 
               or rl in rl_track):
            rl = ng.generateReactionList(ns, nr, realBoundaryIdsInd)
            st = ng.getFullStoichiometryMatrix(rl, ns).tolist()
            stt = ng.removeBoundaryNodes(np.array(st))
        antStr = ng.generateAntimony(realFloatingIds, realBoundaryIds, stt[1],
                                      stt[2], rl, boundary_init=realBoundaryVal)
        try:
            r = te.loada(antStr)

            counts = 0
            countf = 0
            
#            ss = r.steadyStateSolver
#            ss.allow_approx = True
#            ss.allow_presimulation = True
            r.steadyState()
            
            p_bound = ng.generateParameterBoundary(r.getGlobalParameterIds())
            res = scipy.optimize.differential_evolution(f1, args=(r,), 
                               bounds=p_bound, maxiter=optiMaxIter, tol=optiTol,
                               polish=optiPolish, seed=r_seed)
            
            if not res.success:
                numBadModels += 1
            else:
                # TODO: Might be able to cut the bottom part by simply using 
                # the obj func value from optimizer
                r = te.loada(antStr)
                r.setValues(r.getGlobalParameterIds(), res.x)
                    
                r.steadyState()
                SS_i = r.getFloatingSpeciesConcentrations()
                
                # Buggy model
#                if np.any(SS_i > 1e5):
#                    r.reset()
#                    ss.allow_presimulation = True
#                    ss.presimulation_time = 100
#                    r.steadyState()
#                    SS_i = r.getFloatingSpeciesConcentrations()
                        
                if np.any(SS_i < 1e-5) or np.any(SS_i > 1e5):
                    numBadModels += 1
                else:
                    concCC_i = r.getScaledConcentrationControlCoefficientMatrix()
                    
                    if np.isnan(concCC_i).any():
                        numBadModels += 1
                    else:
                        concCC_i[np.abs(concCC_i) < 1e-16] = 0 # Set small values to zero
                        
#                        concCC_i_row = concCC_i.rownames
#                        concCC_i_col = concCC_i.colnames
#                        concCC_i = concCC_i[np.argsort(concCC_i_row)]
#                        concCC_i = concCC_i[:,np.argsort(concCC_i_col)]
                        
#                        count_i = np.array(np.unravel_index(np.argsort(concCC_i, axis=None), concCC_i.shape)).T
                        dist_i = w1*((np.linalg.norm(realConcCC - concCC_i))/
                                      (1 + np.sum(np.equal(np.sign(np.array(realConcCC)), np.sign(np.array(concCC_i))))))
#                        + np.sum(r.getReactionRates() < 0))
#                                    + 1/(1 + np.sum((count_i == realCount).all(axis=1))))
                        
#                        dist_i = w1*(np.linalg.norm(realConcCC - concCC_i))
                        
                        ens_dist[numGoodModels] = dist_i
                        r.reset()
                        ens_model[numGoodModels] = r.getAntimony(current=True)
                        ens_rl[numGoodModels] = rl
                        rl_track.append(rl)
                        
                        numGoodModels = numGoodModels + 1
        except:
            numBadModels = numBadModels + 1
        antimony.clearPreviousLoads()
        numIter = numIter + 1
        if int(numIter/1000) == (numIter/1000):
            print("Number of iterations = " + str(numIter))
        if int(numIter/10000) == (numIter/10000):
            print("Number of good models = " + str(numGoodModels))
    
    print("In generation: 1")
    print("Number of total iterations = " + str(numIter))
    print("Number of bad models = " + str(numBadModels))
    
    return ens_dist, ens_model, ens_rl, rl_track