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
        objFlux = args[0].getReactionRates()
        objFlux[np.abs(objFlux) < 1e-16] = 0 # Set small values to zero
        
        dist_obj = ((np.linalg.norm(args[1] - objCC))/
                    (1 + np.sum(np.equal(np.sign(np.array(args[1])), np.sign(np.array(objCC))))))
                    
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


def initialize(Parameters):
    global countf
    global counts
    
    numBadModels = 0
    numGoodModels = 0
    numIter = 0
    
    ens_dist = np.empty(Parameters.ens_size)
    ens_model = np.empty(Parameters.ens_size, dtype='object')
    ens_rl = np.empty(Parameters.ens_size, dtype='object')
    rl_track = []
    
    # Initial Random generation
    while (numGoodModels < Parameters.ens_size):
        # Ensure no redundant model
        rl = ng.generateReactionList(Parameters)
        st = ng.getFullStoichiometryMatrix(rl, Parameters.ns).tolist()
        stt = ng.removeBoundaryNodes(np.array(st))
        while rl in rl_track:
            rl = ng.generateReactionList(Parameters)
            st = ng.getFullStoichiometryMatrix(rl, Parameters.ns).tolist()
            stt = ng.removeBoundaryNodes(np.array(st))
        antStr = ng.generateAntimony(Parameters.realFloatingIds, Parameters.realBoundaryIds, stt[1],
                                      stt[2], rl, boundary_init=Parameters.realBoundaryVal)
        try:
            r = te.loada(antStr)

            counts = 0
            countf = 0
            
            r.steadyState()
            
            p_bound = ng.generateParameterBoundary(r.getGlobalParameterIds())
            res = scipy.optimize.differential_evolution(f1, args=(r,), 
                               bounds=p_bound, maxiter=Parameters.optiMaxIter, 
                               tol=Parameters.optiTol,
                               polish=Parameters.optiPolish, 
                               seed=Parameters.r_seed)
            
            if not res.success:
                numBadModels += 1
            else:
                # TODO: Might be able to cut the bottom part by simply using 
                # the obj func value from optimizer
                r = te.loada(antStr)
                r.setValues(r.getGlobalParameterIds(), res.x)
                    
                r.steadyState()
                SS_i = r.getFloatingSpeciesConcentrations()
                
                if np.any(SS_i < 1e-5) or np.any(SS_i > 1e5):
                    numBadModels += 1
                else:
                    concCC_i = r.getScaledConcentrationControlCoefficientMatrix()
                    
                    if np.isnan(concCC_i).any():
                        numBadModels += 1
                    else:
                        concCC_i[np.abs(concCC_i) < 1e-16] = 0 # Set small values to zero
                        
                        dist_i = ((np.linalg.norm(Parameters.realConcCC - concCC_i))/
                                      (1 + np.sum(np.equal(np.sign(np.array(Parameters.realConcCC)), np.sign(np.array(concCC_i))))))
                        
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





