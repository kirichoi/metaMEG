# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:11:42 2019

@author: Kiri Choi
"""

import os, sys
import tellurium as te
import roadrunner
import numpy as np
from multiprocessing import Pool, Manager, Process
from concurrent import futures
import time
import core
import ioutils
import analysis
import networkGenerator as ng
import plotting as pt

if __name__ == '__main__':
    roadrunner.Config.setValue(roadrunner.Config.ROADRUNNER_DISABLE_WARNINGS, 3)
#    roadrunner.Config.setValue(roadrunner.Config.LOADSBMLOPTIONS_CONSERVED_MOIETIES, True)


#%% Settings
    
    class Parameters:
        # Input data ==========================================================
        
        INPUT = None
        READ_SETTINGS = None
        
        
        # Test models =========================================================
        
        # 'FFL', 'Linear', 'Nested', 'Branched', 'Central'
        modelType = 'FFL_m_i'
        
        
        # General settings ====================================================
        
        # Number of generations
        n_gen = 10
        # Size of output ensemble
        ens_size = 10
        # Number of models passed on the next generation without mutation
        pass_size = int(ens_size/10)
        # Number of models to mutate
        mut_size = int(ens_size/2)
        # Maximum iteration allowed for random generation
        maxIter_gen = 20
        # Maximum iteration allowed for mutation
        maxIter_mut = 20
        # Set conserved moiety
        conservedMoiety = False
        # Set steadyStateSelections
        steadyStateSelections = None
        
        
        # Optimizer settings ==================================================
        
        # Maximum iteration allowed for optimizer
        optiMaxIter = 100
        optiTol = 1.
        optiPolish = False
        FLUX = False
        
        
        # Random settings =====================================================
        
        # random seed
        r_seed = 123123
        # Flag for adding Gaussian noise to steady-state and control coefficiant values
        NOISE = False
        # Standard deviation of Gaussian noise
        ABS_NOISE_STD = 0.01
        # Standard deviation of Gaussian noise
        REL_NOISE_STD = 0.1
        
        
        # Plotting settings ===================================================
        
        # Flag for plots
        PLOT = True
        # Flag for saving plots
        SAVE_PLOT = True
        
        
        # Data settings =======================================================
        
        # Flag for collecting models
        EXPORT_ALL_MODELS = True
        # Flag for saving collected models
        EXPORT_OUTPUT = True
        # Flag for saving current settings
        EXPORT_SETTINGS = True
        # Path to save the output
        EXPORT_PATH = './USE/output_FFL_i_mp_ref'
        
        # Flag to run algorithm
        RUN = True
        

#%%    
        if conservedMoiety:
            roadrunner.Config.setValue(roadrunner.Config.LOADSBMLOPTIONS_CONSERVED_MOIETIES, True)
        else:
            roadrunner.Config.setValue(roadrunner.Config.LOADSBMLOPTIONS_CONSERVED_MOIETIES, False)

        roadrunner.Config.setValue(roadrunner.Config.STEADYSTATE_APPROX, True)
        roadrunner.Config.setValue(roadrunner.Config.STEADYSTATE_APPROX_MAX_STEPS, 5)
        roadrunner.Config.setValue(roadrunner.Config.STEADYSTATE_APPROX_TIME, 1000000)
#        roadrunner.Config.setValue(roadrunner.Config.STEADYSTATE_APPROX_TOL, 1e-3)
        
        # Using one of the test models
        realModel = ioutils.testModels(modelType)
        
        try:
            realRR = te.loada(realModel)
        except:
            realRR = te.loadSBMLModel(realModel)
        
        realNumBoundary = realRR.getNumBoundarySpecies()
        realNumFloating = realRR.getNumFloatingSpecies() 
        realFloatingIds = np.sort(realRR.getFloatingSpeciesIds())
        realBoundaryIds = np.sort(realRR.getBoundarySpeciesIds())
        allIds = realRR.getFloatingSpeciesIds() + realRR.getBoundarySpeciesIds()
        allIds.sort()
        realFloatingIdsInd = np.searchsorted(allIds, realFloatingIds)
        realBoundaryIdsInd = np.searchsorted(allIds, realBoundaryIds)
        realFloatingVal = realRR.getFloatingSpeciesConcentrations()
        realBoundaryVal = realRR.getBoundarySpeciesConcentrations()
        realGlobalParameterIds = realRR.getGlobalParameterIds()
        
        if steadyStateSelections != None:
            realRR.steadyStateSelections = np.sort(steadyStateSelections)
        else:
            steadyStateSelections = realFloatingIds
            realRR.steadyStateSelections = realFloatingIds
        
        realRR.steadyState()
        realSteadyState = realRR.getFloatingSpeciesConcentrations()
        realSteadyStateRatio = np.divide(realSteadyState, np.min(realSteadyState))
        if FLUX:
            realFlux = realRR.getReactionRates()
        realRR.reset()
        realRR.steadyState()
        if FLUX:
            realFluxCC = realRR.getScaledFluxControlCoefficientMatrix()
        realConcCC = realRR.getScaledConcentrationControlCoefficientMatrix()
        
        if FLUX:
            realFluxCC[np.abs(realFluxCC) < 1e-6] = 0
        realConcCC[np.abs(realConcCC) < 1e-6] = 0
        
        # Ordering
        if FLUX:
            realFluxCCrow = realFluxCC.rownames
            realFluxCCcol = realFluxCC.colnames
            realFluxCC = realFluxCC[np.argsort(realFluxCCrow)]
            realFluxCC = realFluxCC[:,np.argsort(realFluxCCcol)]
        
        realConcCCrow = realConcCC.rownames
        realConcCCcol = realConcCC.colnames
        realConcCC = realConcCC[np.argsort(realConcCCrow)]
        realConcCC = realConcCC[:,np.argsort(realConcCCcol)]
        
        if FLUX:
            realFlux = realFlux[np.argsort(realRR.getReactionIds())]
        
        ns = realNumBoundary + realNumFloating # Number of species
        nr = realRR.getNumReactions() # Number of reactions
        
        realReactionList = ng.generateReactionListFromAntimony(realModel)
        knownReactionList = ng.generateKnownReactionListFromAntimony(realModel)
        compInfo, compVal = ng.generateCompartmentFromAntimony(realModel)
        
        n_range = range(1, n_gen)
        ens_range = range(ens_size)
        mut_range = range(mut_size)
        r_range = range(nr)
        
    
#%%
    
    if Parameters.RUN:    
        
        print("Original Control Coefficients")
        print(Parameters.realConcCC)
        print("Original Steady State Ratio")
        print(Parameters.realSteadyStateRatio)
        
        if Parameters.NOISE:
            for i in range(len(Parameters.realConcCC)):
                for j in range(len(Parameters.realConcCC[i])):
                    Parameters.realConcCC[i][j] = (Parameters.realConcCC[i][j] + 
                              np.random.normal(0,Parameters.ABS_NOISE_STD) +
                              np.random.normal(0,np.abs(Parameters.realConcCC[i][j]*Parameters.REL_NOISE_STD)))
            
            print("Control Coefficients with Noise Added")
            print(Parameters.realConcCC)
    #%%
    
        # Define seed and ranges
        np.random.seed(Parameters.r_seed)
        
        best_dist = []
        avg_dist = []
        med_dist = []
        top5_dist = []
        
    #%%
        
        manager = Manager()
        rl_track = manager.list()
    
        t1 = time.time()
        
        # Initialize
        ens_dist, ens_model, ens_rl, rl_track = core.initialize(Parameters)
        
        dist_top_ind = np.argsort(ens_dist)
        dist_top = ens_dist[dist_top_ind]
        model_top = ens_model[dist_top_ind]
#        rl_top = ens_rl[dist_top_ind]
        
        best_dist.append(dist_top[0])
        avg_dist.append(np.average(dist_top))
        med_dist.append(np.median(dist_top))
        top5_dist.append(np.average(np.unique(dist_top)[:max(int(0.05*Parameters.ens_size), 1)]))
        print("Minimum distance: " + str(best_dist[-1]))
        print("Top 5 distance: " + str(top5_dist[-1]))
        print("Average distance: " + str(avg_dist[-1]))
        
        breakFlag = False
        
        # TODO: Remove for loop
        for n in Parameters.n_range:
            
            minind = np.argsort(ens_dist)[:Parameters.pass_size]
            tarind = np.delete(np.arange(Parameters.ens_size), minind)
            mut_p = 1/ens_dist[tarind]/np.sum(1/ens_dist[tarind])
            mut_ind = np.random.choice(tarind, size=Parameters.mut_size-Parameters.pass_size, 
                                               replace=False, p=mut_p)
            mut_ind = np.append(mut_ind, minind)
            mut_ind_inv = np.setdiff1d(np.arange(Parameters.ens_size), mut_ind)
            
            eval_dist = np.empty(Parameters.mut_size)
            eval_model = np.empty(Parameters.mut_size, dtype='object')
            eval_rl = np.empty(Parameters.mut_size, dtype='object')
            
#            pool = Pool(processes=2)
#            eval_dist[Parameters.mut_range], eval_model[Parameters.mut_range], eval_rl[Parameters.mut_range] = zip(*pool.map(core.mutate_and_evaluate, zip(Parameters,
#                                                              ens_model[mut_ind[Parameters.mut_range]], 
#                                                              ens_dist[mut_ind[Parameters.mut_range]], 
#                                                              ens_rl[mut_ind[Parameters.mut_range]])))
            
            for m in Parameters.mut_range:
                

                eval_dist_m, eval_model_m, eval_rl_m, rl_track = core.mutate_and_evaluate(Parameters,
                                                                              ens_model[mut_ind[m]], 
                                                                              ens_dist[mut_ind[m]], 
                                                                              ens_rl[mut_ind[m]],
                                                                              rl_track)
                eval_dist[m] = eval_dist_m
                eval_model[m] = eval_model_m
                eval_rl[m] = eval_rl_m
#                rl_track = rl_track + rl_track_m
#                
                
            ens_model[mut_ind] = eval_model
            ens_dist[mut_ind] = eval_dist
            ens_rl[mut_ind] = eval_rl
            
            rnd_dist, rnd_model, rnd_rl, rl_track = core.random_gen(Parameters,
                                                          ens_model[mut_ind_inv], 
                                                          ens_dist[mut_ind_inv], 
                                                          ens_rl[mut_ind_inv],
                                                          rl_track)
            ens_model[mut_ind_inv] = rnd_model
            ens_dist[mut_ind_inv] = rnd_dist
            ens_rl[mut_ind_inv] = rnd_rl
            
            dist_top_ind = np.argsort(ens_dist)
            dist_top = ens_dist[dist_top_ind]
            model_top = ens_model[dist_top_ind]
            
            best_dist.append(dist_top[0])
            avg_dist.append(np.average(dist_top))
            med_dist.append(np.median(dist_top))
            top5_dist.append(np.average(np.unique(dist_top)[:max(int(0.05*Parameters.ens_size), 1)]))
            print("In generation: " + str(n + 1))
            print("Minimum distance: " + str(best_dist[-1]))
            print("Top 5 distance: " + str(top5_dist[-1]))
            print("Average distance: " + str(avg_dist[-1]))

        # Check run time
        t2 = time.time()
        print(t2 - t1)
        
        #%%
        # Collect models
        minInd, log_dens, kde_xarr, kde_idx = analysis.selectWithKernalDensity(dist_top)

    #%%
        EXPORT_PATH = os.path.abspath(os.path.join(os.getcwd(), Parameters.EXPORT_PATH))
        
        if Parameters.PLOT:
            # Convergence
            if Parameters.SAVE_PLOT:
                if not os.path.exists(EXPORT_PATH):
                    os.mkdir(EXPORT_PATH)
                if not os.path.exists(os.path.join(EXPORT_PATH, 'images')):
                    os.mkdir(os.path.join(EXPORT_PATH, 'images'))
                pt.plotAllProgress([best_dist, avg_dist, med_dist, top5_dist], 
                                   labels=['Best', 'Avg', 'Median', 'Top 5 percent'],
                                   SAVE_PATH=os.path.join(EXPORT_PATH, 'images/AllConvergences.pdf'))
            else:
                pt.plotAllProgress([best_dist, avg_dist, med_dist, top5_dist], 
                                   labels=['Best', 'Avg', 'Median', 'Top 5 percent'])
            # TODO: Add polishing with fast optimizer 
            
            # Average residual
            if Parameters.SAVE_PLOT:
                pt.plotResidual(Parameters.realModel,
                                ens_model, 
                                ens_dist, 
                                SAVE_PATH=os.path.join(EXPORT_PATH, 'images/average_residual.pdf'))
            else:
                pt.plotResidual(Parameters.realModel, ens_model, ens_dist)
                
            # Distance histogram with KDE
            if Parameters.SAVE_PLOT:
                pt.plotDistanceHistogramWithKDE(dist_top, 
                                                log_dens, 
                                                minInd, 
                                                SAVE_PATH=os.path.join(EXPORT_PATH, 'images/distance_hist_w_KDE.pdf'))
            else:
                pt.plotDistanceHistogramWithKDE(dist_top, log_dens, minInd)
                
            
    #%%
        if Parameters.EXPORT_SETTINGS:
            ioutils.exportSettings(Parameters, path=EXPORT_PATH)
        
        if Parameters.EXPORT_OUTPUT:
            if Parameters.EXPORT_ALL_MODELS:
                model_col = model_top
                dist_col = dist_top
            else:
                model_col = model_top[:kde_idx]
                dist_col = dist_top[:kde_idx]
            ioutils.exportOutputs(model_col, 
                                  dist_col, 
                                  [best_dist, avg_dist, med_dist, top5_dist], 
                                  Parameters,
                                  t2-t1, 
                                  rl_track, 
                                  path=EXPORT_PATH)
