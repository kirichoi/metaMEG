# -*- coding: utf-8 -*-

import tellurium as te
import numpy as np
import tesbml
import copy

class ReactionType:
    UNIUNI = 0
    BIUNI = 1
    UNIBI = 2
    BIBI = 3
    OTHER = 4

class RegulationType:
    DEFAULT = 0
    INHIBITION = 1
    ACTIVATION = 2
    INIHIBITION_ACTIVATION = 3
    
class Reversibility:
    IRREVERSIBLE = 0
    REVERSIBLE = 1
    
class RLP:
    Default = 0.79
    Inhib = 0.1
    Activ = 0.1
    Inhibactiv = 0.01


def pickReactionType(remove=None):

    if remove != None:
        if remove == RegulationType.DEFAULT:
            # preg = max(0, preg-0.3) 
            Inhib = RLP.Inhib + RLP.Default/2.1 #0.48*(1-preg)
            Activ = RLP.Activ + RLP.Default/2.1 #0.48*(1-preg)
            # Inhibactiv = 0.02*(1-preg)
            Default = 0 #1 - Inhib - Activ - Inhibactiv
        elif remove == RegulationType.INHIBITION:
            Default = RLP.Default + RLP.Inhib/3
            Inhib = 0
            Activ = RLP.Activ + RLP.Inhib/3
        elif remove == RegulationType.ACTIVATION:
            Default = RLP.Default + RLP.Activ/3
            Inhib = RLP.Inhib + RLP.Activ/3
            Activ = 0
        elif remove == RegulationType.INIHIBITION_ACTIVATION:
            Default = RLP.Default + RLP.Inhibactiv/3
            Inhib = RLP.Inhib + RLP.Inhibactiv/3
            Activ = RLP.Activ + RLP.Inhibactiv/3
    else:
        Default = RLP.Default
        Inhib = RLP.Inhib
        Activ = RLP.Activ
    
    rt = np.random.random()    
        
    if rt < Default:
        regType = RegulationType.DEFAULT
    elif (rt >= Default) and (rt < Default + Inhib):
        regType = RegulationType.INHIBITION
    elif (rt >= Default + Inhib) and (rt < Default + Inhib + Activ):
        regType = RegulationType.ACTIVATION
    else:
        regType = RegulationType.INIHIBITION_ACTIVATION
    
    return regType


# Generates a reaction network in the form of a reaction list
# reactionList = [nSpecies, reaction, ....]
# reaction = [reactionType, [list of reactants], [list of product], rateConstant]
def generateReactionList(Parameters):
    
    reactionList = copy.deepcopy(Parameters.knownReactionList)
    
    for r_idx in range(Parameters.nr):
        rct_id = np.array(reactionList[r_idx][3])[:,1]
        prd_id = np.array(reactionList[r_idx][4])[:,1]
        
        regType = pickReactionType()
        
        if regType == RegulationType.DEFAULT:
            act_id = []
            inhib_id = []
        elif regType == RegulationType.INHIBITION:
            act_id = []
            delList = np.concatenate([rct_id, prd_id])
            delList = np.unique(np.append(delList, Parameters.realBoundaryIdsInd))
            cList = np.delete(np.arange(Parameters.ns), delList)
            if len(cList) == 0:
                inhib_id = []
                regType = RegulationType.DEFAULT
            else:
                inhib_id = np.unique(np.random.choice(cList, size=np.random.choice([1,2,3], p=[0.89, 0.1, 0.01]))).tolist()
        elif regType == RegulationType.ACTIVATION:
            inhib_id = []
            delList = np.concatenate([rct_id, prd_id])
            delList = np.unique(np.append(delList, Parameters.realBoundaryIdsInd))
            cList = np.delete(np.arange(Parameters.ns), delList)
            if len(cList) == 0:
                act_id = []
                regType = RegulationType.DEFAULT
            else:
                act_id = np.unique(np.random.choice(cList, size=np.random.choice([1,2,3], p=[0.89, 0.1, 0.01]))).tolist()
        else:
            delList = np.concatenate([rct_id, prd_id])
            delList = np.unique(np.append(delList, Parameters.realBoundaryIdsInd))
            cList = np.delete(np.arange(Parameters.ns), delList)
            if len(cList) < 2:
                act_id = []
                inhib_id = []
                regType = RegulationType.DEFAULT
            else:
                reg_id = np.random.choice(cList, size=2, replace=False)
                act_id = [reg_id[0]]
                inhib_id = [reg_id[1]]
                    
        reactionList[r_idx][1] = regType
        reactionList[r_idx][5] = act_id
        reactionList[r_idx][6] = inhib_id

    return reactionList
    

def generateMutation(Parameters, rl, model):

    reactionList = copy.deepcopy(rl)
    
    r = te.loada(model)
    r.steadyStateApproximate()
    
    concCC = r.getScaledConcentrationControlCoefficientMatrix()
    
    cFalse = (1 + np.sum(np.not_equal(np.sign(np.array(Parameters.realConcCC)), 
                                      np.sign(np.array(concCC))), axis=0))
    
    tempdiff = cFalse*np.linalg.norm(Parameters.realConcCC - concCC, axis=0)
    
    r_idx = np.random.choice(np.arange(Parameters.nr), p=np.divide(tempdiff,np.sum(tempdiff)))
    rct_id = np.array(reactionList[r_idx][3])[:,1]
    prd_id = np.array(reactionList[r_idx][4])[:,1]
    
    # preg = np.divide(np.count_nonzero(np.array(reactionList)[:,1]), Parameters.nr)
    regType = pickReactionType(reactionList[r_idx][1])
    
    if regType == RegulationType.DEFAULT:
        act_id = []
        inhib_id = []
    elif regType == RegulationType.INHIBITION:
        act_id = []
        delList = np.concatenate([rct_id, prd_id])
        delList = np.unique(np.append(delList, Parameters.realBoundaryIdsInd))
        cList = np.delete(np.arange(Parameters.ns), delList)
        if len(cList) == 0:
            inhib_id = []
            regType = RegulationType.DEFAULT
        else:
            inhib_id = np.unique(np.random.choice(cList, size=np.random.choice([1,2,3], p=[0.89, 0.1, 0.01]))).tolist()
    elif regType == RegulationType.ACTIVATION:
        inhib_id = []
        delList = np.concatenate([rct_id, prd_id])
        delList = np.unique(np.append(delList, Parameters.realBoundaryIdsInd))
        cList = np.delete(np.arange(Parameters.ns), delList)
        if len(cList) == 0:
            act_id = []
            regType = RegulationType.DEFAULT
        else:
            act_id = np.unique(np.random.choice(cList, size=np.random.choice([1,2,3], p=[0.89, 0.1, 0.01]))).tolist()
    else:
        delList = np.concatenate([rct_id, prd_id])
        delList = np.unique(np.append(delList, Parameters.realBoundaryIdsInd))
        cList = np.delete(np.arange(Parameters.ns), delList)
        if len(cList) < 2:
            act_id = []
            inhib_id = []
            regType = RegulationType.DEFAULT
        else:
            reg_id = np.random.choice(cList, size=2, replace=False)
            act_id = [reg_id[0]]
            inhib_id = [reg_id[1]]
        
    reactionList[r_idx][1] = regType
    reactionList[r_idx][5] = act_id
    reactionList[r_idx][6] = inhib_id

    return reactionList



# Include boundary and floating species
# Returns a list:
# [New Stoichiometry matrix, list of floatingIds, list of boundaryIds]
def getFullStoichiometryMatrix(reactionList, ns):
    rlcopy = copy.deepcopy(reactionList)
    st = np.zeros((ns, len(rlcopy)))
    
    for index, rind in enumerate(rlcopy):
        for i in range(len(rlcopy[index][3])):
            reactant = rlcopy[index][3][i][1]
            st[reactant, index] = st[reactant, index] - int(rlcopy[index][3][i][0])
        for j in range(len(rlcopy[index][4])):
            product = rlcopy[index][4][j][1]
            st[product, index] = st[product, index] + int(rlcopy[index][4][j][0])

    return st
        

# Removes boundary or orphan species from stoichiometry matrix
def removeBoundaryNodes(st):
    
    dims = st.shape
    
    nSpecies = dims[0]
    nReactions = dims[1]
    
    speciesIds = np.arange(nSpecies)
    indexes = []
    orphanSpecies = []
    countBoundarySpecies = 0
    for r in range(nSpecies): 
        # Scan across the columns, count + and - coefficients
        plusCoeff = 0
        minusCoeff = 0
        for c in range(nReactions):
            if st[r,c] < 0:
                minusCoeff += 1
            elif st[r,c] > 0:
                plusCoeff += 1
        if plusCoeff == 0 and minusCoeff == 0:
            # No reaction attached to this species
            orphanSpecies.append(r)
        elif plusCoeff == 0 and minusCoeff != 0:
            # Species is a source
            indexes.append(r)
            countBoundarySpecies += 1
        elif minusCoeff == 0 and plusCoeff != 0:
            # Species is a sink
            indexes.append(r)
            countBoundarySpecies += 1

    floatingIds = np.delete(speciesIds, indexes+orphanSpecies, axis=0).astype('int')
    floatingIds = floatingIds.tolist()

    boundaryIds = indexes
    return [np.delete(st, indexes + orphanSpecies, axis=0), floatingIds, boundaryIds]


def generateReactionListFromAntimony(antStr):
    """
    """
    import sympy
    
    try:
        r = te.loada(antStr)
    except:
        r = te.loadSBMLModel(antStr)
    
    numBnd = r.getNumBoundarySpecies()
    numFlt = r.getNumFloatingSpecies()
    boundaryId = r.getBoundarySpeciesIds()
    floatingId = r.getFloatingSpeciesIds()
    allId = floatingId + boundaryId
    # allId.sort()
    nr = r.getNumReactions()
    
    # prepare symbols for sympy
    boundaryId_sympy = [] 
    floatingId_sympy = []
    
    # Fix issues with reserved characters
    for i in range(numBnd):
        if boundaryId[i] == 'S':
            boundaryId_sympy.append('_S')
        else:
            boundaryId_sympy.append(boundaryId[i])
    
    for i in range(numFlt):
        if floatingId[i] == 'S':
            floatingId_sympy.append('_S')
        else:
            floatingId_sympy.append(floatingId[i])
    
    paramIdsStr = ' '.join(r.getGlobalParameterIds())
    floatingIdsStr = ' '.join(floatingId_sympy)
    boundaryIdsStr = ' '.join(boundaryId_sympy)
    comparmentIdsStr = ' '.join(r.getCompartmentIds())
    
    allIds = paramIdsStr + ' ' + floatingIdsStr + ' ' + boundaryIdsStr + ' ' + comparmentIdsStr
    
    avsym = sympy.symbols(allIds)
    
    # extract reactant, product, modifiers, and kinetic laws
    rct = []
    rctst = []
    prd = []
    prdst = []
    mod = []
    r_type = []
    kineticLaw = []
    mod_type = []
    
    doc = tesbml.readSBMLFromString(r.getSBML())
    sbmlmodel = doc.getModel()

    for slr in sbmlmodel.getListOfReactions():
        temprct = []
        temprctst = []
        tempprd = []
        tempprdst = []
        tempmod = []
        
        sbmlreaction = sbmlmodel.getReaction(slr.getId())
        for sr in range(sbmlreaction.getNumReactants()):
            sbmlrct = sbmlreaction.getReactant(sr)
            temprct.append(sbmlrct.getSpecies())
            temprctst.append(int(sbmlrct.getStoichiometry()))
        for sp in range(sbmlreaction.getNumProducts()):
            sbmlprd = sbmlreaction.getProduct(sp)
            tempprd.append(sbmlprd.getSpecies())
            tempprdst.append(int(sbmlprd.getStoichiometry()))
        for sm in range(sbmlreaction.getNumModifiers()):
            sbmlmod = sbmlreaction.getModifier(sm)
            tempmod.append(sbmlmod.getSpecies())
        kl = sbmlreaction.getKineticLaw()
        
        rct.append(temprct)
        rctst.append(temprctst)
        prd.append(tempprd)
        prdst.append(tempprdst)
        mod.append(tempmod)
        
        # Update kinetic law according to change in species name
        kl_split = kl.getFormula().split(' ')
        for i in range(len(kl_split)):
            if kl_split[i] == 'S':
                kl_split[i] = '_S'
        
        kineticLaw.append(' '.join(kl_split))
    
    # use sympy for analyzing modifiers weSmart
    for ml in range(len(mod)):
        mod_type_temp = []
        expression = kineticLaw[ml]
        n,d = sympy.fraction(expression)
        for ml_i in range(len(mod[ml])):
            sym = sympy.symbols(mod[ml][ml_i])
            if n.has(sym) and not d.has(sym):
                mod_type_temp.append('activator')
            elif d.has(sym) and not n.has(sym):
                mod_type_temp.append('inhibitor')
            elif n.has(sym) and d.has(sym):
                mod_type_temp.append('inhibitor_activator')
            else:
                mod_type_temp.append('modifier')
        mod_type.append(mod_type_temp)
        
        # In case all products are in rate law, assume it is a reversible reaction
        if all(ext in str(n) for ext in prd[ml]):
            r_type.append('reversible')
        else:
            r_type.append('irreversible')
        
    reactionList = []
    
    for i in range(nr):
        inhib = []
        activ = []
        rct_temp = []
        prd_temp = []
        
        if len(rct[i]) == 1:
            if len(prd[i]) == 1:
                rType = 0
            elif len(prd[i]) == 2:
                rType = 2
            else:
                rType = 4
        elif len(rct[i]) == 2:
            if len(prd[i]) == 1:
                rType = 1
            elif len(prd[i]) == 2:
                rType = 3
            else:
                rType = 4
        else:
            rType = 4
        
        for j in range(len(rct[i])):
            rct_temp.append((rctst[i][j], allId.index(rct[i][j])))
            
        for j in range(len(prd[i])):
            prd_temp.append((prdst[i][j], allId.index(prd[i][j])))
        
        if len(mod_type[i]) == 0:
            regType = 0
        else:
            for k in range(len(mod_type[i])):
                if mod_type[i][k] == 'inhibitor':
                    inhib.append(allId.index(mod[i][k]))
                elif mod_type[i][k] == 'activator':
                    activ.append(allId.index(mod[i][k]))
                
                if len(inhib) > 0:
                    if len(activ) == 0:
                        regType = 1
                    else:
                        regType = 3
                else:
                    regType = 2
                
        if r_type[i] == 'reversible':
            revType = 1
        else:
            revType = 0

        reactionList.append([rType, 
                             regType, 
                             revType, 
                             rct_temp, 
                             prd_temp,
                             activ,
                             inhib])
    
    return reactionList


def generateKnownReactionListFromAntimony(antStr):
    """
    """
    import sympy
    
    try:
        r = te.loada(antStr)
    except:
        r = te.loadSBMLModel(antStr)
    
    numBnd = r.getNumBoundarySpecies()
    numFlt = r.getNumFloatingSpecies()
    boundaryId = r.getBoundarySpeciesIds()
    floatingId = r.getFloatingSpeciesIds()
    allId = floatingId + boundaryId
    # allId.sort()
    nr = r.getNumReactions()
    
    # prepare symbols for sympy
    boundaryId_sympy = [] 
    floatingId_sympy = []
    
    # Fix issues with reserved characters
    for i in range(numBnd):
        if boundaryId[i] == 'S':
            boundaryId_sympy.append('_S')
        else:
            boundaryId_sympy.append(boundaryId[i])
    
    for i in range(numFlt):
        if floatingId[i] == 'S':
            floatingId_sympy.append('_S')
        else:
            floatingId_sympy.append(floatingId[i])
    
    paramIdsStr = ' '.join(r.getGlobalParameterIds())
    floatingIdsStr = ' '.join(floatingId_sympy)
    boundaryIdsStr = ' '.join(boundaryId_sympy)
    comparmentIdsStr = ' '.join(r.getCompartmentIds())
    
    allIds = paramIdsStr + ' ' + floatingIdsStr + ' ' + boundaryIdsStr + ' ' + comparmentIdsStr
    
    avsym = sympy.symbols(allIds)
    
    # extract reactant, product, modifiers, and kinetic laws
    rct = []
    rctst = []
    prd = []
    prdst = []
    mod = []
    r_type = []
    kineticLaw = []
    mod_type = []
    
    doc = tesbml.readSBMLFromString(r.getSBML())
    sbmlmodel = doc.getModel()

    for slr in sbmlmodel.getListOfReactions():
        temprct = []
        temprctst = []
        tempprd = []
        tempprdst = []
        tempmod = []
        
        sbmlreaction = sbmlmodel.getReaction(slr.getId())
        for sr in range(sbmlreaction.getNumReactants()):
            sbmlrct = sbmlreaction.getReactant(sr)
            temprct.append(sbmlrct.getSpecies())
            temprctst.append(int(sbmlrct.getStoichiometry()))
        for sp in range(sbmlreaction.getNumProducts()):
            sbmlprd = sbmlreaction.getProduct(sp)
            tempprd.append(sbmlprd.getSpecies())
            tempprdst.append(int(sbmlprd.getStoichiometry()))
        for sm in range(sbmlreaction.getNumModifiers()):
            sbmlmod = sbmlreaction.getModifier(sm)
            tempmod.append(sbmlmod.getSpecies())
        kl = sbmlreaction.getKineticLaw()
        
        rct.append(temprct)
        rctst.append(temprctst)
        prd.append(tempprd)
        prdst.append(tempprdst)
        mod.append(tempmod)
        
        # Update kinetic law according to change in species name
        kl_split = kl.getFormula().split(' ')
        for i in range(len(kl_split)):
            if kl_split[i] == 'S':
                kl_split[i] = '_S'
        
        kineticLaw.append(' '.join(kl_split))

    # use sympy for analyzing modifiers weSmart
    for ml in range(len(mod)):
        mod_type_temp = []
        expression = kineticLaw[ml]
        n,d = sympy.fraction(expression)
        for ml_i in range(len(mod[ml])):
            sym = sympy.symbols(mod[ml][ml_i])
            if n.has(sym) and not d.has(sym):
                mod_type_temp.append('activator')
            elif d.has(sym) and not n.has(sym):
                mod_type_temp.append('inhibitor')
            elif n.has(sym) and d.has(sym):
                mod_type_temp.append('inhibitor_activator')
            else:
                mod_type_temp.append('modifier')
        mod_type.append(mod_type_temp)
        
        # In case all products are in rate law, assume it is a reversible reaction
        if all(ext in str(n) for ext in prd[ml]) and all(ext in prd[ml] for ext in str(n)):
            r_type.append('reversible')
        else:
            r_type.append('irreversible')
        
    reactionList = []
    
    for i in range(nr):
        inhib = []
        activ = []
        rct_temp = []
        prd_temp = []
        
        if len(rct[i]) == 1:
            if len(prd[i]) == 1:
                rType = 0
            elif len(prd[i]) == 2:
                rType = 2
            else:
                rType = 4
        elif len(rct[i]) == 2:
            if len(prd[i]) == 1:
                rType = 1
            elif len(prd[i]) == 2:
                rType = 3
            else:
                rType = 4
        else:
            rType = 4
        
        for j in range(len(rct[i])):
            rct_temp.append((rctst[i][j], allId.index(rct[i][j])))
            
        for j in range(len(prd[i])):
            prd_temp.append((prdst[i][j], allId.index(prd[i][j])))
        
        regType = 0
                
        if r_type[i] == 'reversible':
            revType = 1
        else:
            revType = 0

        reactionList.append([rType, 
                             regType, 
                             revType, 
                             rct_temp, 
                             prd_temp,
                             activ,
                             inhib])
    
    return reactionList



def generateCompartmentFromAntimony(antStr):
    """
    """
    
    try:
        r = te.loada(antStr)
    except:
        r = te.loadSBMLModel(antStr)
    
    compInfo = []
    compVal = []
    
    doc = tesbml.readSBMLFromString(r.getSBML())
    sbmlmodel = doc.getModel()
    
    for sl in sbmlmodel.getListOfSpecies():
        s = sbmlmodel.getSpecies(sl.getId())
        sid = s.getId()
        sc = s.getCompartment()
        
        compInfo.append([sid, sc])
    
    for cpl in sbmlmodel.getListOfCompartments():
        cp = sbmlmodel.getCompartment(cpl.getId())
        compVal.append([cpl.getId(), cp.getVolume()])
        
    return compInfo, compVal



def generateSimpleRateLaw(rl, allId, Jind, simple=True):
    
    Klist = []
    
    T = ''
    D = ''
    ACT = ''
    INH = ''
    
    # T
    if len(rl[Jind][3]) == 0 or len(rl[Jind][4]) == 0:
        rateLaw = 'Kf' + str(Jind)
        Klist.append('Kf' + str(Jind))
    else:
        if simple:
            T = T + 'Kf' + str(Jind) + '*('
        else:
            T = T + '(Kf' + str(Jind) + '*'
        Klist.append('Kf' + str(Jind))
        
        for i in range(len(rl[Jind][3])):
            T = T + '(' + str(allId[rl[Jind][3][i][1]])
            if rl[Jind][3][i][0] != 1:
                T = T + '^' + str(rl[Jind][3][i][0])
            T = T + ')'
            if i < len(rl[Jind][3]) - 1:
                T = T + '*'
        
        if rl[Jind][2] == Reversibility.REVERSIBLE:
            if simple:
                T = T + ' - '
            else:
                T = T + ' - Kr' + str(Jind) + '*'
                Klist.append('Kr' + str(Jind))
                
            for i in range(len(rl[Jind][4])):
                T = T + '(' + str(allId[rl[Jind][4][i][1]])
                if rl[Jind][4][i][0] != 1:
                    T = T + '^' + str(rl[Jind][4][i][0])
                T = T + ')'
                if i < len(rl[Jind][4]) - 1:
                    T = T + '*'
                
        T = T + ')'
            
        # D
        # D = D + '1 + '
        
        # for i in range(len(rl[Jind][3])):
        #     D = D + '(' + str(allId[rl[Jind][3][i][1]])
        #     if rl[Jind][3][i][0] != 1:
        #         D = D + '^' + str(rl[Jind][3][i][0])
        #     D = D + ')'
        #     if i < len(rl[Jind][3]) - 1:
        #         D = D + '*'
        
        # if rl[Jind][2] == Reversibility.REVERSIBLE:
        #     D = D + ' + '
        #     for i in range(len(rl[Jind][4])):
        #         D = D + '(' + str(allId[rl[Jind][4][i][1]])
        #         if rl[Jind][4][i][0] != 1:
        #             D = D + '^' + str(rl[Jind][4][i][0])
        #         D = D + ')'
        #         if i < len(rl[Jind][4]) - 1:
        #             D = D + '*'
        
        # Activation
        if (len(rl[Jind][5]) > 0):
            for i in range(len(rl[Jind][5])):
                ACT = ACT + '(1 + Ka' + str(Jind) + str(i) + '*'
                Klist.append('Ka' + str(Jind) + str(i))
                ACT = ACT + str(allId[rl[Jind][5][i]]) + ')*'
                
        # Inhibition
        if (len(rl[Jind][6]) > 0):
            for i in range(len(rl[Jind][6])):
                INH = INH + '(1/(1 + Ki' + str(Jind) + str(i) + '*'
                Klist.append('Ki' + str(Jind) + str(i))
                INH = INH + str(allId[rl[Jind][6][i]]) + '))*'
        
        rateLaw = ACT + INH + T# + '/(' + D + ')'
        
    return rateLaw, Klist


def generateAntimony(fids, bids, allIds, rl, floating_init=None, boundary_init=None, compInfo=None, compVal=None):
    rlcopy = copy.deepcopy(rl)
    Klist = []
    
    # List species
    antStr = ''
    if len (fids) > 0:
        antStr = antStr + 'var ' + str(fids[0])
        for index in fids[1:]:
            antStr = antStr + ', ' + str(index)
        antStr = antStr + ';\n'
    
    if len (bids) > 0:
        antStr = antStr + 'const ' + str(bids[0])
        for index in bids[1:]:
            antStr = antStr + ', ' + str(index)
        antStr = antStr + ';\n'
        
    if compInfo:
        for i in range(len(compInfo)):
            if compInfo[i][1] != 'default_compartment':
                antStr = antStr + 'species ' + str(compInfo[i][0]) + ' in ' + str(compInfo[i][1]) + '; '
        antStr = antStr + '\n'
    
    # List reactions
    for index, rind in enumerate(rlcopy):
#        if rind[0] == ReactionType.UNIUNI:
            # UniUni
        antStr = antStr + 'J' + str(index) + ': '
        for i in range(len(rlcopy[index][3])):
            if i != 0:
                antStr = antStr + ' + '
            if type(rlcopy[index][3][i]) == tuple:
                if rlcopy[index][3][i][0] == 1:
                    antStr = antStr + str(allIds[rlcopy[index][3][i][1]])
                else:
                    antStr = antStr + str(rlcopy[index][3][i][0]) + str(allIds[rlcopy[index][3][i][1]])
            else:
                antStr = antStr + str(allIds[rlcopy[index][3][i]])
        antStr = antStr + ' -> '
        for j in range(len(rlcopy[index][4])):
            if j != 0:
                antStr = antStr + ' + '
            if type(rlcopy[index][4][j]) == tuple:
                if rlcopy[index][4][j][0] == 1:
                    antStr = antStr + str(allIds[rlcopy[index][4][j][1]])
                else:
                    antStr = antStr + str(rlcopy[index][4][j][0]) + str(allIds[rlcopy[index][4][j][1]])
            else:
                antStr = antStr + str(allIds[rlcopy[index][4][j]])
        antStr = antStr + '; '
        RateLaw, klist_i = generateSimpleRateLaw(rl, allIds, index)
        antStr = antStr + RateLaw
        Klist.append(klist_i)
        antStr = antStr + ';\n'

    if compVal:
        for i in range(len(compVal)):
            if compVal[i][0] != 'default_compartment':
                antStr = antStr + str(compVal[i][0]) + ' = ' + str(compVal[i][1]) + ';\n'
    
    # List rate constants
    antStr = antStr + '\n'
    Klist_f = [item for sublist in Klist for item in sublist]
    
    for i in range(len(Klist_f)):
        if Klist_f[i].startswith('Kf'):
            antStr = antStr + Klist_f[i] + ' = 1\n'
        elif Klist_f[i].startswith('Kr'):
            antStr = antStr + Klist_f[i] + ' = 0.5\n'
        elif Klist_f[i].startswith('Ka'):
            antStr = antStr + Klist_f[i] + ' = 0.1\n'
        elif Klist_f[i].startswith('Ki'):
            antStr = antStr + Klist_f[i] + ' = 0.1\n'
        
    # Initialize boundary species
    antStr = antStr + '\n'
    if type(boundary_init) == type(None):
        for index, bind in enumerate(bids):
            antStr = antStr + str(bind) + ' = ' + str(np.random.randint (1,6)) + '\n'
    else:
        for index, bind in enumerate(bids):
            antStr = antStr + str(bind) + ' = ' + str(boundary_init[index]) + '\n'
    
    # Initialize floating species
    if type(boundary_init) == type(None):
        for index, find in enumerate(fids):
            antStr = antStr + str(find) + ' = ' + str(np.random.random()) + '\n'
    else:
        for index, find in enumerate(fids):
            antStr = antStr + str(find) + ' = ' + str(floating_init[index]) + '\n'
        
    return antStr
     

def generateParameterBoundary(glgp):
    
    pBound = []
    
    for i in range(len(glgp)):
        if glgp[i].startswith('Kf'):
            pBound.append((1e-1, 10.))
        elif glgp[i].startswith('Kr'):
            pBound.append((1e-1, 10.))
        elif glgp[i].startswith('Ka'):
            pBound.append((1e-3, 10.))
        elif glgp[i].startswith('Ki'):
            pBound.append((1e-3, 10.))

    return pBound
    

    
