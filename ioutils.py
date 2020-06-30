# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 17:18:02 2018

@author: ckiri
"""

import sys, os
import numpy as np
import pandas as pd

def exportSettings(Parameters, path=None):
    """
    Export settings to specified path
    
    :param Parameters: Parameters object
    :param path: path to save file
    """
    
    if path:
        outputdir = path
    else:
        outputdir = os.path.join(os.getcwd(), 'output')
    
    outputtxt = open(os.path.join(outputdir, 'settings.txt'), 'w')
    outputtxt.writelines('------------------------- SETTINGS -------------------------\n')
    outputtxt.writelines('No. of Generations: ' + str(Parameters.n_gen) + '\n')
    outputtxt.writelines('Ensemble Size: ' + str(Parameters.ens_size) + '\n')
    
    outputtxt.writelines('Pass Size: ' + str(Parameters.pass_size) + '\n')
    outputtxt.writelines('Mut. Size: ' + str(Parameters.mut_size) + '\n')
    outputtxt.writelines('Random Gen. Max. Iter: ' + str(Parameters.maxIter_gen) + '\n')
    outputtxt.writelines('Max. Iter. Mut.: ' + str(Parameters.maxIter_mut) + '\n')
    outputtxt.writelines('Conserved Moiety: ' + str(Parameters.conservedMoiety) + '\n')
    outputtxt.writelines('Steady State Sel.: ' + str(Parameters.steadyStateSelections) + '\n')
    outputtxt.writelines('Opti. Max. Iter.: ' + str(Parameters.optiMaxIter) + '\n')
    outputtxt.writelines('Opti. Tol.: ' + str(Parameters.optiTol) + '\n')
    outputtxt.writelines('Opti. Polish: ' + str(Parameters.optiPolish) + '\n')
    outputtxt.writelines('Use Flux: ' + str(Parameters.FLUX) + '\n')
    outputtxt.writelines('Random Seed: ' + str(Parameters.r_seed) + '\n')
    outputtxt.writelines('Add Noise: ' + str(Parameters.NOISE) + '\n')
    outputtxt.writelines('Noise Abs. Std.: ' + str(Parameters.ABS_NOISE_STD) + '\n')
    outputtxt.writelines('Noise Rel. Std.: ' + str(Parameters.REL_NOISE_STD) + '\n')
    outputtxt.close()
    
    
def exportOutputs(models, dists, minInd, dist_list, Parameters, time, rl_track, path=None):
    """
    Export all outputs to specified path
    
    :param models: list of collected models
    :param dists: list of collected model distances
    :param dist_list: list of best, average, median, and top 5 distances 
    :param Parameters: Parameters object
    :param time: excution time
    :param rl_track: list of analyzed topologies
    :param path: path to save file
    """
    
    if path:
        outputdir = path
    else:
        outputdir = os.path.join(os.getcwd(), 'output')
        
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)
    if not os.path.exists(os.path.join(outputdir, 'models')):
        os.mkdir(os.path.join(outputdir, 'models'))
    
    df = pd.DataFrame(np.array(dists), columns=['distance'])
    df.to_csv(os.path.join(outputdir, 'dist_collected.txt'))
    
    stat = pd.DataFrame(np.array(dist_list).T, 
                        columns=['generation best',
                                 'generation average',
                                 'generation median',
                                 'generation top5'])
    stat.to_csv(os.path.join(outputdir, 'dist_stat.txt'))
    
    outputtxt = open(os.path.join(outputdir, 'report.txt'), 'w')
    outputtxt.writelines('------------------------- REPORT -------------------------\n')
    outputtxt.writelines('RUN COMPLETE. HERE ARE SOME METRIC YOU MIGHT BE INTERESTED\n')
    outputtxt.writelines('No. of Generations: ' + str(Parameters.n_gen) + '\n')
    outputtxt.writelines('Ensemble Size: ' + str(Parameters.ens_size) + '\n')
    outputtxt.writelines('No. of Collected Models: ' + str(len(models)) + '\n')
    outputtxt.writelines('Ensemble Size: ' + str(minInd[0]) + '\n')
    outputtxt.writelines('Run Time: ' + str(time) + ' s\n')
    outputtxt.writelines('No. Stoich. Analyzed: ' + str(len(rl_track)) + '\n')
    outputtxt.close()
    
    for i in range(len(models)):
        modeltxt = open(os.path.join(outputdir, 'models/model_%03d' % i + '.txt'), 'w')
        modeltxt.write(models[i])
        modeltxt.close()
    

def importSettings(settingsPath):
    """
    """


def importModels(modelsPath):
    """
    Read model files
    
    :param modelsPath: path to a directory containing model files
    :returns: list of model strings
    """
    
    modelfiles = [f for f in os.listdir(modelsPath) if os.path.isfile(os.path.join(modelsPath, f))]

    antstr = []
    for i in modelfiles:
        sbmlstr = open(os.path.join(modelsPath, i), 'r')
        antstr.append(sbmlstr.read())
        sbmlstr.close()
    
    return antstr


def importData(dataPath):
    """
    Read data encoded in csv
    
    :param dataPath: path to a csv file
    :returns: DataFrame
    """
    
    if os.path.exists(dataPath):
        df = pd.read_csv(dataPath)
        return df
    else:
        raise Exception("Cannot find the file at the specified path")
    
    
def testModels(modelType):
    """
    Returns a test model
    
    :param modelType: model name, e.g. 'FFL', 'Linear', 'Nested', 'Branched'
    :returns: Antimony string
    """
    
    if modelType == 'Linear_m':
        # Linear    
        realModel = """
        var S1, S2, S3, S4;
        const S0, S5;
        J0: S0 -> S1; Kf0*S0/(1 + S0);
        J1: S1 -> S2; Kf1*S1/(1 + S1);
        J2: S2 -> S3; Kf2*S2/(1 + S2);
        J3: S3 -> S4; Kf3*S3/(1 + S3);
        J4: S4 -> S5; Kf4*S4/(1 + S4);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.27503984992
       
        S0 = 3
        S5 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        S4 = 1
        """
    elif modelType == 'Linear_m_i':
        realModel = """
        var S1, S2, S3, S4;
        const S0, S5;
        J0: S0 -> S1; Kf0*S0/(1 + S0)*1/(1 + Ki0*S4);;
        J1: S1 -> S2; Kf1*S1/(1 + S1);
        J2: S2 -> S3; Kf2*S2/(1 + S2);
        J3: S3 -> S4; Kf3*S3/(1 + S3);
        J4: S4 -> S5; Kf4*S4/(1 + S4);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.27503984992
        Ki0 = 0.974569278466
       
        S0 = 3
        S5 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        S4 = 1
        """
    elif modelType == 'Linear_r':
        # Linear    
        realModel = """
        var S1, S2, S3, S4;
        const S0, S5;
        J0: S0 -> S1; (Kf0*S0 - Kr0*S1)/(1 + S0 + S1);
        J1: S1 -> S2; (Kf1*S1 - Kr1*S2)/(1 + S1 + S2);
        J2: S2 -> S3; (Kf2*S2 - Kr2*S3)/(1 + S2 + S3);
        J3: S3 -> S4; (Kf3*S3 - Kr3*S4)/(1 + S3 + S4);
        J4: S4 -> S5; (Kf4*S4 - Kr4*S5)/(1 + S4 + S5);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.27503984992
        Kr0 = 0.072790656829
        Kr1 = 0.27828563882
        Kr2 = 0.166906190765
        Kr3 = 0.27057924345
        Kr4 = 0.1026847763
       
        S0 = 3
        S5 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        S4 = 1
        """
    elif modelType == 'FFL_m':
        # FFL    
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; Kf0*S0/(1 + S0);
        J1: S1 -> S2; Kf1*S1/(1 + S1);
        J2: S2 -> S3; Kf2*S2/(1 + S2);
        J3: S3 -> S4; Kf3*S3/(1 + S3);
        J4: S1 -> S3; Kf4*S1/(1 + S1);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.148522702962
       
        S0 = 3
        S4 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        """
    elif modelType == 'FFL_m_a':
        # FFL    
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; Kf0*S0/(1 + S0);
        J1: S1 -> S2; Kf1*S1/(1 + S1);
        J2: S2 -> S3; Kf2*S2/(1 + S2);
        J3: S3 -> S4; Kf3*S3/(1 + S3);
        J4: S1 -> S3; (Kf4*S1/(1 + S1))*(1 + Ka0*S2);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.148522702962
        Ka0 = 0.883848629231
       
        S0 = 3
        S4 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        """
    elif modelType == 'FFL_m_i':
        # I1FFL
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; Kf0*S0/(1 + S0);
        J1: S1 -> S2; Kf1*S1/(1 + S1);
        J2: S2 -> S3; Kf2*S2/(1 + S2);
        J3: S3 -> S4; Kf3*S3/(1 + S3);
        J4: S1 -> S3; (Kf4*S1/(1 + S1))*1/(1 + Ki0*S2);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.148522702962
        Ki0 = 0.974569278466
       
        S0 = 3
        S4 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        """
    elif modelType == 'FFL_r':
        # FFL    
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; (Kf0*S0 - Kr0*S1)/(1 + S0 + S1);
        J1: S1 -> S2; (Kf0*S1 - Kr1*S2)/(1 + S1 + S2);
        J2: S2 -> S3; (Kf2*S2 - Kr2*S3)/(1 + S2 + S3);
        J3: S3 -> S4; (Kf3*S3 - Kr3*S4)/(1 + S3 + S4);
        J4: S1 -> S3; (Kf4*S1 - Kr4*S3)/(1 + S1 + S3);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.548522702962
        Kr0 = 0.072790656829
        Kr1 = 0.27828563882
        Kr2 = 0.166906190765
        Kr3 = 0.27057924345
        Kr4 = 0.1026847763
       
        S0 = 3
        S4 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        """
    elif modelType == 'FFL_r_a':
        # FFL    
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; (Kf0*S0 - Kr0*S1)/(1 + S0 + S1);
        J1: S1 -> S2; (Kf0*S1 - Kr1*S2)/(1 + S1 + S2);
        J2: S2 -> S3; (Kf2*S2 - Kr2*S3)/(1 + S2 + S3);
        J3: S3 -> S4; (Kf3*S3 - Kr3*S4)/(1 + S3 + S4);
        J4: S1 -> S3; (1 + Ka0*S2)*(Kf4*S1 - Kr4*S3)/(1 + S1 + S3);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.548522702962
        Kr0 = 0.072790656829
        Kr1 = 0.27828563882
        Kr2 = 0.166906190765
        Kr3 = 0.27057924345
        Kr4 = 0.1026847763
        Ka0 = 0.883848629231
       
        S0 = 3
        S4 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        """
    elif modelType == 'FFL_r_i':
        # FFL    
        realModel = """
        var S1, S2, S3;
        const S0, S4;
        J0: S0 -> S1; (Kf0*S0 - Kr0*S1)/(1 + S0 + S1);
        J1: S1 -> S2; (Kf0*S1 - Kr1*S2)/(1 + S1 + S2);
        J2: S2 -> S3; (Kf2*S2 - Kr2*S3)/(1 + S2 + S3);
        J3: S3 -> S4; (Kf3*S3 - Kr3*S4)/(1 + S3 + S4);
        J4: S1 -> S3; 1/(1 + Ki0*S2)*(Kf4*S1 - Kr4*S3)/(1 + S1 + S3);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.548522702962
        Kr0 = 0.072790656829
        Kr1 = 0.27828563882
        Kr2 = 0.166906190765
        Kr3 = 0.27057924345
        Kr4 = 0.1026847763
        Ki0 = 0.974569278466
       
        S0 = 3
        S4 = 5
        S1 = 1
        S2 = 1
        S3 = 1
        """
    elif modelType == 'SigPath':
        realModel = """
        var S0, S1, S2, S3, S4, S5;
        J0: S0 -> S1; 1/(1 + S5/Ki0)*(Kf0*S0)/(1 + S0);
        J1: S1 -> S0; (Kf1*S1)/(1 + S1);
        J2: S2 -> S3; (1 + Ka2*S1)*(Kf2*S2)/(1 + S2);
        J3: S3 -> S2; (Kf3*S3)/(1 + S3);
        J4: S4 -> S5; (1 + Ka4*S3)*(Kf4*S4)/(1 + S4);
        J5: S5 -> S4; (Kf5*S5)/(1 + S5);
        
        Kf0 = 0.485822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.948522702962
        Kf5 = 0.272790656829
        Ki0 = 0.34569278466
        Ka2 = 0.6276983967
        Ka4 = 0.1143526464
        
        S0 = 1
        S1 = 1
        S2 = 1
        S3 = 1
        S4 = 1
        S5 = 1
        """
    elif modelType == 'SigPath_Bare':
        realModel = """
        var S0, S1, S2, S3, S4, S5;
        J0: S0 -> S1; (Kf0*S0)/(1 + S0);
        J1: S1 -> S0; (Kf1*S1)/(1 + S1);
        J2: S2 -> S3; (Kf2*S2)/(1 + S2);
        J3: S3 -> S2; (Kf3*S3)/(1 + S3);
        J4: S4 -> S5; (Kf4*S4)/(1 + S4);
        J5: S5 -> S4; (Kf5*S5)/(1 + S5);
        
        Kf0 = 0.485822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.948522702962
        Kf5 = 0.272790656829
        
        S0 = 1
        S1 = 1
        S2 = 1
        S3 = 1
        S4 = 1
        S5 = 1
        """
    elif modelType == 'NameTest':
        realModel = """
        var X, Y, Z;
        const INP, SNK;
        J0: INP -> X; Kf0*INP/(1 + INP);
        J1: X -> Y; Kf1*X/(1 + X);
        J2: Y -> Z; Kf2*Y/(1 + Y);
        J3: Z -> SNK; Kf3*Z/(1 + Z);
        J4: X -> Z; Kf4*X/(1 + X);
       
        Kf0 = 0.285822003905
        Kf1 = 0.571954691013
        Kf2 = 0.393173236422
        Kf3 = 0.75830845241
        Kf4 = 0.148522702962
       
        INP = 3
        SNK = 5
        X = 1
        Y = 1
        Z = 1
        """
    elif modelType == 'GLYCO':
        realModel = r"./models/MODEL1505110000_ant_mod.txt"
    else:
        realModel = modelType
        
    return realModel
