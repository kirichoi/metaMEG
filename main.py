# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 17:11:42 2019

@author: Kiri Choi
"""

import os, sys
import tellurium as te
import roadrunner
import numpy as np
from multiprocessing import Pool
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
        # model = 'FFL_m_i'
        model = r'D:\Archive\OneDrive\Models\SBML Models\sauro_demo_mod_irrev.xml'
        
        
        # General settings ====================================================
        
        # Number of generations
        n_gen = 500
        # Size of output ensemble
        ens_size = 200
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
        optiATol = 1.
        optiPolish = False
        FLUX = False
        workers = 1
        
        
        # RNG and noise settings ==============================================
        
        # Random seed
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
        # Flag for plotting the best fitted network model
        PLOT_NETWORK = True
        
        
        # Data settings =======================================================
        
        # Flag for saving collected models
        EXPORT_OUTPUT = True
        # Flag for collecting all models
        EXPORT_ALL_MODELS = True
        # Flag for saving current settings
        EXPORT_SETTINGS = True
        # Path to the output
        EXPORT_PATH = './USE/output_sauro_demo_irrev'
        
        # Flag to run the algorithm
        RUN = False
        

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
        realModel = ioutils.testModels(model)
        
        try:
            realRR = te.loada(realModel)
        except:
            realRR = te.loadSBMLModel(realModel)
        
        realNumBoundary = realRR.getNumBoundarySpecies()
        realNumFloating = realRR.getNumFloatingSpecies() 
        # realFloatingIds = np.sort(realRR.getFloatingSpeciesIds())
        # realBoundaryIds = np.sort(realRR.getBoundarySpeciesIds())
        realFloatingIds = realRR.getFloatingSpeciesIds()
        realBoundaryIds = realRR.getBoundarySpeciesIds()
        allIds = realRR.getFloatingSpeciesIds() + realRR.getBoundarySpeciesIds()
        # allIds.sort()
        realFloatingIdsInd = np.arange(realNumFloating)#np.searchsorted(allIds, realFloatingIds)
        realBoundaryIdsInd = np.arange(realNumFloating, realNumFloating+realNumBoundary)#np.searchsorted(allIds, realBoundaryIds)
        realFloatingVal = realRR.getFloatingSpeciesConcentrations()#[np.argsort(realRR.getFloatingSpeciesIds())]
        realBoundaryVal = realRR.getBoundarySpeciesConcentrations()#[np.argsort(realRR.getBoundarySpeciesIds())]
        realGlobalParameterIds = realRR.getGlobalParameterIds()
        
        if steadyStateSelections != None:
            realRR.steadyStateSelections = steadyStateSelections#np.sort(steadyStateSelections)
        else:
            steadyStateSelections = realFloatingIds
            realRR.steadyStateSelections = realFloatingIds
        
        realRR.steadyState()
        realSteadyState = realRR.getFloatingSpeciesConcentrations()
        realSteadyStateRatio = np.divide(realSteadyState, np.min(realSteadyState))
        realFlux = realRR.getReactionRates()
        realRR.reset()
        realRR.steadyState()
        realFluxCC = realRR.getScaledFluxControlCoefficientMatrix()
        realConcCC = realRR.getScaledConcentrationControlCoefficientMatrix()
        
        realFluxCC[np.abs(realFluxCC) < 1e-12] = 0
        realConcCC[np.abs(realConcCC) < 1e-12] = 0
        
        # Ordering
        # if FLUX:
        #     realFluxCCrow = realFluxCC.rownames
        #     realFluxCCcol = realFluxCC.colnames
        #     # realFluxCC = realFluxCC[np.argsort(realFluxCCrow)]
        #     realFluxCC = realFluxCC[:,np.argsort(realFluxCCcol)]
        
        # realConcCCrow = realConcCC.rownames
        # realConcCCcol = realConcCC.colnames
        # realConcCC = realConcCC[np.argsort(realConcCCrow)]
        # realConcCC = realConcCC[:,np.argsort(realConcCCcol)]
        
        # if FLUX:
        #     realFlux = realFlux[np.argsort(realRR.getReactionIds())]
        
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
            minidx = np.argsort(ens_dist)[:Parameters.pass_size]
            taridx = np.delete(np.arange(Parameters.ens_size), minidx)
            mut_p = 1/ens_dist[taridx]/np.sum(1/ens_dist[taridx])
            mutidx = np.random.choice(taridx, size=Parameters.mut_size-Parameters.pass_size, 
                                               replace=False, p=mut_p)
            mutidx = np.append(mutidx, minidx)
            mutidx_inv = np.setdiff1d(np.arange(Parameters.ens_size), mutidx)
            
            eval_dist, eval_model, eval_rl, rl_track = core.mutate_and_evaluate(Parameters,
                                                                      ens_model[mutidx], 
                                                                      ens_dist[mutidx], 
                                                                      ens_rl[mutidx],
                                                                      rl_track)
            ens_model[mutidx] = eval_model
            ens_dist[mutidx] = eval_dist
            ens_rl[mutidx] = eval_rl
            
            rnd_dist, rnd_model, rnd_rl, rl_track = core.random_gen(Parameters,
                                                          ens_model[mutidx_inv], 
                                                          ens_dist[mutidx_inv], 
                                                          ens_rl[mutidx_inv],
                                                          rl_track)
            ens_model[mutidx_inv] = rnd_model
            ens_dist[mutidx_inv] = rnd_dist
            ens_rl[mutidx_inv] = rnd_rl
            
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
        minInd, log_dens = analysis.selectWithKernalDensity(dist_top)

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
                                                minInd, 
                                                SAVE_PATH=os.path.join(EXPORT_PATH, 'images/distance_hist_w_KDE.pdf'))
            else:
                pt.plotDistanceHistogramWithKDE(dist_top, minInd)
            
            if Parameters.PLOT_NETWORK:
                if Parameters.SAVE_PLOT:
                    pt.plotNetworkComparison(Parameters.realModel, 
                                             model_top[0], 
                                             scale=1,
                                             SAVE_PATH=os.path.join(EXPORT_PATH, 'images/net_comp_real_best.pdf'))
                    pt.plotWeightedNetwork(Parameters.realModel, 
                                           model_top[:minInd[0]], 
                                           scale=2., 
                                           threshold=0.25,
                                           SAVE_PATH=os.path.join(EXPORT_PATH, 'images/net_comp_real_ensemble.pdf'))
                else:
                    pt.plotNetworkComparison(Parameters.realModel, 
                                             model_top[0], 
                                             scale=1)
                    pt.plotWeightedNetwork(Parameters.realModel, 
                                           model_top[:minInd[0]], 
                                           scale=2., 
                                           threshold=0.25)
                
            
    #%%
        if Parameters.EXPORT_SETTINGS:
            ioutils.exportSettings(Parameters, path=EXPORT_PATH)
        
        if Parameters.EXPORT_OUTPUT:
            if Parameters.EXPORT_ALL_MODELS:
                model_col = model_top
                dist_col = dist_top
            else:
                model_col = model_top[:minInd[0]]
                dist_col = dist_top[:minInd[0]]
            ioutils.exportOutputs(model_col, 
                                  dist_col, 
                                  minInd, 
                                  [best_dist, avg_dist, med_dist, top5_dist], 
                                  Parameters, 
                                  t2-t1, 
                                  rl_track, 
                                  path=EXPORT_PATH)
