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
        args[0].steadyStateApproximate()
        objCCC = args[0].getScaledConcentrationControlCoefficientMatrix()
        objCCC[np.abs(objCCC) < 1e-12] = 0 # Set small values to zero
        if np.isnan(objCCC).any():
            dist_obj = 10000
        else:
            if args[3]:
                objFlux = args[0].getReactionRates()
                objFlux[np.abs(objFlux) < 1e-12] = 0 # Set small values to zero
    #        objFCC = args[0].getScaledFluxControlCoefficientMatrix()
    #        objFCC[np.abs(objFCC) < 1e-12] = 0 # Set small values to zero
            
            objCCC_row = objCCC.rownames
            objCCC_col = objCCC.colnames
            objCCC = objCCC[np.argsort(objCCC_row)]
            objCCC = objCCC[:,np.argsort(objCCC_col)]
            
            if args[3]:
                objFlux = objFlux[np.argsort(objCCC_col)]
            
                dist_obj = (((np.linalg.norm(args[1] - objCCC)) + (np.linalg.norm(args[2] - objFlux))) *
                            ((1 + np.sum(np.equal(np.sign(np.array(args[1])), np.sign(np.array(objCCC))))) +
                            (1 + np.sum(np.equal(np.sign(np.array(args[2])), np.sign(np.array(objFlux)))))))
            else:
                dist_obj = ((np.linalg.norm(args[1] - objCCC))*(1 + 
                             np.sum(np.not_equal(np.sign(np.array(args[1])), 
                                                 np.sign(np.array(objCCC))))))
                
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
    rl_track.append(Parameters.knownReactionList)
    
    # Initial Random generation
    while (numGoodModels < Parameters.ens_size):
        # Ensure no redundant model
        rl = ng.generateReactionList(Parameters)
        while rl in rl_track:
            rl = ng.generateReactionList(Parameters)
        antStr = ng.generateAntimony(Parameters.realFloatingIds, 
                                     Parameters.realBoundaryIds, 
                                     Parameters.allIds,
                                     rl, 
                                     boundary_init=Parameters.realBoundaryVal,
                                     floating_init=Parameters.realFloatingVal,
                                     compInfo=Parameters.compInfo,
                                     compVal=Parameters.compVal)
        try:
            r = te.loada(antStr)
            r.steadyStateSelections = Parameters.steadyStateSelections

            counts = 0
            countf = 0
            
            r.steadyStateApproximate()
            
            p_bound = ng.generateParameterBoundary(r.getGlobalParameterIds())
            res = scipy.optimize.differential_evolution(f1, 
                                                        args=(r, Parameters.realConcCC, Parameters.realFlux, Parameters.FLUX), 
                                                        bounds=p_bound, 
                                                        maxiter=Parameters.optiMaxIter, 
                                                        tol=Parameters.optiTol,
                                                        polish=Parameters.optiPolish, 
                                                        seed=Parameters.r_seed)
            
            if not res.success:
                numBadModels += 1
            else:
                # TODO: Might be able to cut the bottom part by simply using 
                # the obj func value from optimizer
                r = te.loada(antStr)
                r.steadyStateSelections = Parameters.steadyStateSelections
                r.setValues(r.getGlobalParameterIds(), res.x)
                
                r.steadyStateApproximate()
                SS_i = r.getFloatingSpeciesConcentrations()
                
                r.steadyStateApproximate()
                
                if np.any(SS_i < 1e-5) or np.any(SS_i > 1e5):
                    numBadModels += 1
                else:
                    concCC_i = r.getScaledConcentrationControlCoefficientMatrix()
                    if Parameters.FLUX:
                        flux_i = r.getReactionRates()
        
                    if np.isnan(concCC_i).any():
                        numBadModels += 1
                    else:
                        concCC_i[np.abs(concCC_i) < 1e-12] = 0 # Set small values to zero
                        if Parameters.FLUX:
                            flux_i[np.abs(flux_i) < 1e-12] = 0 # Set small values to zero
                        
                        concCC_i_row = concCC_i.rownames
                        concCC_i_col = concCC_i.colnames
                        concCC_i = concCC_i[np.argsort(concCC_i_row)]
                        concCC_i = concCC_i[:,np.argsort(concCC_i_col)]
                        
                        if Parameters.FLUX:
                            flux_i = flux_i[np.argsort(concCC_i_col)]
                        
                            dist_i = (((np.linalg.norm(Parameters.realConcCC - concCC_i)) + 
                                      (np.linalg.norm(Parameters.realFlux - flux_i))) * 
                                    ((1 + np.sum(np.not_equal(np.sign(np.array(Parameters.realConcCC)), 
                                                              np.sign(np.array(concCC_i))))) + 
                                     (1 + np.sum(np.not_equal(np.sign(np.array(Parameters.realFlux)), 
                                                              np.sign(np.array(flux_i)))))))
                        else:
                            dist_i = ((np.linalg.norm(Parameters.realConcCC - concCC_i))*(1 + 
                                       np.sum(np.not_equal(np.sign(np.array(Parameters.realConcCC)), 
                                                           np.sign(np.array(concCC_i))))))
                        
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


def mutate_and_evaluate(Parameters, listantStr, listdist, listrl, rl_track):
    global countf
    global counts
    
    eval_dist = np.empty(Parameters.mut_size)
    eval_model = np.empty(Parameters.mut_size, dtype='object')
    eval_rl = np.empty(Parameters.mut_size, dtype='object')
    
    for m in Parameters.mut_range:
        o = 0
        
        rl = ng.generateMutation(Parameters, listrl[m], listantStr[m])
        
        while ((rl in rl_track) and (o < Parameters.maxIter_mut)):
            rl = ng.generateMutation(Parameters, listrl[m], listantStr[m])
            o += 1
        
        if o >= Parameters.maxIter_mut:
            eval_dist[m] = listdist[m]
            eval_model[m] = listantStr[m]
            eval_rl[m] = listrl[m]
        else:
            antStr = ng.generateAntimony(Parameters.realFloatingIds, 
                                         Parameters.realBoundaryIds, 
                                         Parameters.allIds,
                                         rl, 
                                         boundary_init=Parameters.realBoundaryVal,
                                         floating_init=Parameters.realFloatingVal,
                                         compInfo=Parameters.compInfo,
                                         compVal=Parameters.compVal)
            try:
                r = te.loada(antStr)
                r.steadyStateSelections = Parameters.steadyStateSelections
                
                r.steadyStateApproximate()
                
                p_bound = ng.generateParameterBoundary(r.getGlobalParameterIds())
                res = scipy.optimize.differential_evolution(f1, 
                                                            args=(r, Parameters.realConcCC, Parameters.realFlux, Parameters.FLUX), 
                                                            bounds=p_bound, 
                                                            maxiter=Parameters.optiMaxIter, 
                                                            tol=Parameters.optiTol,
                                                            polish=Parameters.optiPolish,
                                                            seed=Parameters.r_seed)
                if not res.success:
                    eval_dist[m] = listdist[m]
                    eval_model[m] = listantStr[m]
                    eval_rl[m] = listrl[m]
                else:
                    r = te.loada(antStr)
                    r.steadyStateSelections = Parameters.steadyStateSelections
                    r.setValues(r.getGlobalParameterIds(), res.x)
                    
                    r.steadyStateApproximate()
                    SS_i = r.getFloatingSpeciesConcentrations()
                    
                    r.steadyStateApproximate()
                    
                    if np.any(SS_i < 1e-5) or np.any(SS_i > 1e5):
                        eval_dist[m] = listdist[m]
                        eval_model[m] = listantStr[m]
                        eval_rl[m] = listrl[m]
                    else:
                        concCC_i = r.getScaledConcentrationControlCoefficientMatrix()
                        if Parameters.FLUX:
                            flux_i = r.getReactionRates()
                        
                        if np.isnan(concCC_i).any():
                            eval_dist[m] = listdist[m]
                            eval_model[m] = listantStr[m]
                            eval_rl[m] = listrl[m]
                        else:
                            concCC_i[np.abs(concCC_i) < 1e-12] = 0 # Set small values to zero
                            if Parameters.FLUX:
                                flux_i[np.abs(flux_i) < 1e-12] = 0 # Set small values to zero
                            
                            concCC_i_row = concCC_i.rownames
                            concCC_i_col = concCC_i.colnames
                            concCC_i = concCC_i[np.argsort(concCC_i_row)]
                            concCC_i = concCC_i[:,np.argsort(concCC_i_col)]
                            
                            if Parameters.FLUX:
                                flux_i = flux_i[np.argsort(concCC_i_col)]
                            
                                dist_i = (((np.linalg.norm(Parameters.realConcCC - concCC_i)) + 
                                           (np.linalg.norm(Parameters.realFlux - flux_i))) * 
                                        ((1 + np.sum(np.not_equal(np.sign(np.array(Parameters.realConcCC)), 
                                                                  np.sign(np.array(concCC_i))))) + 
                                         (1 + np.sum(np.not_equal(np.sign(np.array(Parameters.realFlux)),
                                                                  np.sign(np.array(flux_i)))))))
                            else:
                                dist_i = ((np.linalg.norm(Parameters.realConcCC - concCC_i))*(1 + 
                                           np.sum(np.not_equal(np.sign(np.array(Parameters.realConcCC)), 
                                                               np.sign(np.array(concCC_i))))))
                            
                            if dist_i < listdist[m]:
                                eval_dist[m] = dist_i
                                r.reset()
                                eval_model[m] = r.getAntimony(current=True)
                                eval_rl[m] = rl
                                rl_track.append(rl)
                            else:
                                eval_dist[m] = listdist[m]
                                eval_model[m] = listantStr[m]
                                eval_rl[m] = listrl[m]
            except:
                eval_dist[m] = listdist[m]
                eval_model[m] = listantStr[m]
                eval_rl[m] = listrl[m]
        antimony.clearPreviousLoads()

    return eval_dist, eval_model, eval_rl, rl_track


def random_gen(Parameters, listAntStr, listDist, listrl, rl_track):
    global countf
    global counts
    
    rndSize = len(listDist)
    
    rnd_dist = np.empty(rndSize)
    rnd_model = np.empty(rndSize, dtype='object')
    rnd_rl = np.empty(rndSize, dtype='object')
    
    for l in range(rndSize):
        d = 0
        
        rl = ng.generateReactionList(Parameters)
        # Ensure no redundant models
        while ((rl in rl_track) and (d < Parameters.maxIter_gen)):
            rl = ng.generateReactionList(Parameters)
            d += 1
            
        if d >= Parameters.maxIter_gen:
            rnd_dist[l] = listDist[l]
            rnd_model[l] = listAntStr[l]
            rnd_rl[l] = listrl[l]
        else:
            antStr = ng.generateAntimony(Parameters.realFloatingIds, 
                                         Parameters.realBoundaryIds, 
                                         Parameters.allIds,
                                         rl,
                                         boundary_init=Parameters.realBoundaryVal,
                                         floating_init=Parameters.realFloatingVal,
                                         compInfo=Parameters.compInfo,
                                         compVal=Parameters.compVal)
            try:
                r = te.loada(antStr)
                r.steadyStateSelections = Parameters.steadyStateSelections
                
                r.steadyStateApproximate()
                
                p_bound = ng.generateParameterBoundary(r.getGlobalParameterIds())
                res = scipy.optimize.differential_evolution(f1, 
                                                            args=(r, Parameters.realConcCC, Parameters.realFlux, Parameters.FLUX), 
                                                            bounds=p_bound, 
                                                            maxiter=Parameters.optiMaxIter, 
                                                            tol=Parameters.optiTol,
                                                            polish=Parameters.optiPolish, 
                                                            seed=Parameters.r_seed)
                # Failed to find solution
                if not res.success:
                    rnd_dist[l] = listDist[l]
                    rnd_model[l] = listAntStr[l]
                    rnd_rl[l] = listrl[l]
                else:
                    r = te.loada(antStr)
                    r.steadyStateSelections = Parameters.steadyStateSelections
                    r.setValues(r.getGlobalParameterIds(), res.x)
                    
                    r.steadyStateApproximate()
                    SS_i = r.getFloatingSpeciesConcentrations()
                    
                    r.steadyStateApproximate()
                    
                    if np.any(SS_i < 1e-5) or np.any(SS_i > 1e5):
                        rnd_dist[l] = listDist[l]
                        rnd_model[l] = listAntStr[l]
                        rnd_rl[l] = listrl[l]
                    else:
                        concCC_i = r.getScaledConcentrationControlCoefficientMatrix()
                        if Parameters.FLUX:
                            flux_i = r.getReactionRates()
                        
                        if np.isnan(concCC_i).any():
                            rnd_dist[l] = listDist[l]
                            rnd_model[l] = listAntStr[l]
                            rnd_rl[l] = listrl[l]
                        else:
                            concCC_i[np.abs(concCC_i) < 1e-12] = 0 # Set small values to zero
                            if Parameters.FLUX:
                                flux_i[np.abs(flux_i) < 1e-12] = 0 # Set small values to zero
                            
                            concCC_i_row = concCC_i.rownames
                            concCC_i_col = concCC_i.colnames
                            concCC_i = concCC_i[np.argsort(concCC_i_row)]
                            concCC_i = concCC_i[:,np.argsort(concCC_i_col)]
                            
                            if Parameters.FLUX:
                                flux_i = flux_i[np.argsort(concCC_i_col)]
                            
                                dist_i = (((np.linalg.norm(Parameters.realConcCC - concCC_i)) + 
                                           (np.linalg.norm(Parameters.realFlux - flux_i))) * 
                                        ((1 + np.sum(np.not_equal(np.sign(np.array(Parameters.realConcCC)), 
                                                                  np.sign(np.array(concCC_i))))) + 
                                         (1 + np.sum(np.not_equal(np.sign(np.array(Parameters.realFlux)), 
                                                                  np.sign(np.array(flux_i)))))))
                            else:
                                dist_i = ((np.linalg.norm(Parameters.realConcCC - concCC_i))*(1 + 
                                           np.sum(np.not_equal(np.sign(np.array(Parameters.realConcCC)), 
                                                               np.sign(np.array(concCC_i))))))
                                
                            if dist_i < listDist[l]:
                                rnd_dist[l] = dist_i
                                r.reset()
                                rnd_model[l] = r.getAntimony(current=True)
                                rnd_rl[l] = rl
                                rl_track.append(rl)
                            else:
                                rnd_dist[l] = listDist[l]
                                rnd_model[l] = listAntStr[l]
                                rnd_rl[l] = listrl[l]
            except:
                rnd_dist[l] = listDist[l]
                rnd_model[l] = listAntStr[l]
                rnd_rl[l] = listrl[l]
        antimony.clearPreviousLoads()
    
    return rnd_dist, rnd_model, rnd_rl, rl_track



