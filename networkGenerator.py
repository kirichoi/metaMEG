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

class RegulationType:
    DEFAULT = 0
    INHIBITION = 1
    ACTIVATION = 2
    INIHIBITION_ACTIVATION = 3
    
class Reversibility:
    IRREVERSIBLE = 0
    REVERSIBLE = 1
    
class RLP:
    Default = 0.73
    Inhib = 0.12
    Activ = 0.12
    Inhibactiv = 0.03


def pickReactionType(remove=None):

    if remove != None:
        if remove == RegulationType.DEFAULT:
            Default = 0
            Inhib = RLP.Inhib + RLP.Default/3
            Activ = RLP.Activ + RLP.Default/3
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
        rct_id = reactionList[r_idx][3]
        prd_id = reactionList[r_idx][4]
        
        regType = pickReactionType()
        
        if regType == RegulationType.DEFAULT:
            act_id = []
            inhib_id = []
        elif regType == RegulationType.INHIBITION:
            act_id = []
            delList = np.unique(np.concatenate([rct_id, prd_id]))
            delList = np.append(delList, list(map(str, Parameters.realBoundaryIdsInd)))
            cList = np.delete(np.arange(Parameters.ns), delList)
            if len(cList) == 0:
                inhib_id = []
                regType = RegulationType.DEFAULT
            else:
                inhib_id = np.random.choice(cList, size=np.random.randint(1,3)).tolist()
        elif regType == RegulationType.ACTIVATION:
            inhib_id = []
            delList = np.unique(np.concatenate([rct_id, prd_id]))
            delList = np.append(delList, list(map(str, Parameters.realBoundaryIdsInd)))
            cList = np.delete(np.arange(Parameters.ns), delList)
            if len(cList) == 0:
                act_id = []
                regType = RegulationType.DEFAULT
            else:
                act_id = np.random.choice(cList, size=np.random.randint(1,3)).tolist()
        else:
            delList = np.unique(np.concatenate([rct_id, prd_id]))
            delList = np.append(delList, list(map(str, Parameters.realBoundaryIdsInd)))
            cList = np.delete(np.arange(Parameters.ns), delList)
            if len(cList) == 0:
                act_id = []
                inhib_id = []
                regType = RegulationType.DEFAULT
            else:
                reg_id = np.random.choice(cList, size=2)
                if reg_id[0] == reg_id[1]:
                    act_id = []
                    inhib_id = []
                    regType = RegulationType.DEFAULT
                else:
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
    
    cFalse = (1 + 
              concCC.shape[0] - 
              np.sum(np.equal(np.sign(np.array(Parameters.realConcCC)), 
                              np.sign(np.array(concCC))), axis=0))
    
    tempdiff = cFalse*np.max(np.abs(Parameters.realConcCC - concCC), axis=0)
    
    r_idx = np.random.choice(np.arange(Parameters.nr), p=np.divide(tempdiff,np.sum(tempdiff)))
    rct_id = reactionList[r_idx][3]
    prd_id = reactionList[r_idx][4]
    
    regType = pickReactionType(reactionList[r_idx][1])
    
    if regType == RegulationType.DEFAULT:
        act_id = []
        inhib_id = []
    elif regType == RegulationType.INHIBITION:
        act_id = []
        delList = np.unique(np.concatenate([rct_id, prd_id]))
        delList = np.append(delList, list(map(str, Parameters.realBoundaryIdsInd)))
        cList = np.delete(np.arange(Parameters.ns), delList)
        if len(cList) == 0:
            inhib_id = []
            regType = RegulationType.DEFAULT
        else:
            inhib_id = np.random.choice(cList, size=np.random.randint(1,3)).tolist()
    elif regType == RegulationType.ACTIVATION:
        inhib_id = []
        delList = np.unique(np.concatenate([rct_id, prd_id]))
        delList = np.append(delList, list(map(str, Parameters.realBoundaryIdsInd)))
        cList = np.delete(np.arange(Parameters.ns), delList)
        if len(cList) == 0:
            act_id = []
            regType = RegulationType.DEFAULT
        else:
            act_id = np.random.choice(cList, size=np.random.randint(1,3)).tolist()
    else:
        delList = np.unique(np.concatenate([rct_id, prd_id]))
        delList = np.append(delList, list(map(str, Parameters.realBoundaryIdsInd)))
        cList = np.delete(np.arange(Parameters.ns), delList)
        if len(cList) == 0:
            act_id = []
            inhib_id = []
            regType = RegulationType.DEFAULT
        else:
            reg_id = np.random.choice(cList, size=2)
            if reg_id[0] == reg_id[1]:
                act_id = []
                inhib_id = []
                regType = RegulationType.DEFAULT
            else:
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
    reactionListCopy = copy.deepcopy(reactionList)
    st = np.zeros((ns, len(reactionListCopy)))
    
    for index, rind in enumerate(reactionListCopy):
        if rind[0] == ReactionType.UNIUNI:
            # UniUni
            reactant = reactionListCopy[index][3][0]
            st[reactant, index] = st[reactant, index] - 1
            product = reactionListCopy[index][4][0]
            st[product, index] = st[product, index] + 1
     
        elif rind[0] == ReactionType.BIUNI:
            # BiUni
            reactant1 = reactionListCopy[index][3][0]
            st[reactant1, index] = st[reactant1, index] - 1
            reactant2 = reactionListCopy[index][3][1]
            st[reactant2, index] = st[reactant2, index] - 1
            product = reactionListCopy[index][4][0]
            st[product, index] = st[product, index] + 1

        elif rind[0] == ReactionType.UNIBI:
            # UniBi
            reactant1 = reactionListCopy[index][3][0]
            st[reactant1, index] = st[reactant1, index] - 1
            product1 = reactionListCopy[index][4][0]
            st[product1, index] = st[product1, index] + 1
            product2 = reactionListCopy[index][4][1]
            st[product2, index] = st[product2, index] + 1

        else:
            # BiBi
            reactant1 = reactionListCopy[index][3][0]
            st[reactant1, index] = st[reactant1, index] - 1
            reactant2 = reactionListCopy[index][3][1]
            st[reactant2, index] = st[reactant2, index] - 1
            product1 = reactionListCopy[index][4][0]
            st[product1, index] = st[product1, index] + 1
            product2 = reactionListCopy[index][4][1]
            st[product2, index] = st[product2, index] + 1

    return st
        

# Removes boundary or orphan species from stoichiometry matrix
def removeBoundaryNodes(st):
    
    dims = st.shape
    
    nSpecies = dims[0]
    nReactions = dims[1]
    
    speciesIds = np.arange (nSpecies)
    indexes = []
    orphanSpecies = []
    countBoundarySpecies = 0
    for r in range(nSpecies): 
        # Scan across the columns, count + and - coefficients
        plusCoeff = 0; minusCoeff = 0
        for c in range(nReactions):
            if st[r,c] < 0:
                minusCoeff = minusCoeff + 1
            elif st[r,c] > 0:
                plusCoeff = plusCoeff + 1
        if plusCoeff == 0 and minusCoeff == 0:
            # No reaction attached to this species
            orphanSpecies.append (r)
        elif plusCoeff == 0 and minusCoeff != 0:
            # Species is a source
            indexes.append (r)
            countBoundarySpecies = countBoundarySpecies + 1
        elif minusCoeff == 0 and plusCoeff != 0:
            # Species is a sink
            indexes.append (r)
            countBoundarySpecies = countBoundarySpecies + 1

    floatingIds = np.delete(speciesIds, indexes+orphanSpecies, axis=0).astype('int')
    floatingIds = floatingIds.tolist()

    boundaryIds = indexes
    return [np.delete(st, indexes + orphanSpecies, axis=0), floatingIds, boundaryIds]


def generateReactionListFromAntimony(antStr):
    """
    """
    import sympy
    
    r = te.loada(antStr)
    
    numBnd = r.getNumBoundarySpecies()
    numFlt = r.getNumFloatingSpecies()
    boundaryId = r.getBoundarySpeciesIds()
    floatingId = r.getFloatingSpeciesIds()
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
    prd = []
    mod = []
    r_type = []
    kineticLaw = []
    mod_type = []
    
    doc = tesbml.readSBMLFromString(r.getSBML())
    sbmlmodel = doc.getModel()

    for slr in sbmlmodel.getListOfReactions():
        temprct = []
        tempprd = []
        tempmod = []
        
        sbmlreaction = sbmlmodel.getReaction(slr.getId())
        for sr in range(sbmlreaction.getNumReactants()):
            sbmlrct = sbmlreaction.getReactant(sr)
            temprct.append(sbmlrct.getSpecies())
        for sp in range(sbmlreaction.getNumProducts()):
            sbmlprd = sbmlreaction.getProduct(sp)
            tempprd.append(sbmlprd.getSpecies())
        for sm in range(sbmlreaction.getNumModifiers()):
            sbmlmod = sbmlreaction.getModifier(sm)
            tempmod.append(sbmlmod.getSpecies())
        kl = sbmlreaction.getKineticLaw()
        
        rct.append(sorted(temprct, key=lambda v: (v.upper(), v[0].islower())))
        prd.append(sorted(tempprd, key=lambda v: (v.upper(), v[0].islower())))
        mod.append(sorted(tempmod, key=lambda v: (v.upper(), v[0].islower())))
        
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
            if n.has(mod[ml][ml_i]) and not d.has(mod[ml][ml_i]):
                mod_type_temp.append('activator')
            elif d.has(mod[ml][ml_i]) and not n.has(mod[ml][ml_i]):
                mod_type_temp.append('inhibitor')
            elif n.has(mod[ml][ml_i]) and d.has(mod[ml][ml_i]):
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
        elif len(rct[i]) == 2:
            if len(prd[i]) == 1:
                rType = 1
            elif len(prd[i]) == 2:
                rType = 3
        
        for j in range(len(rct[i])):
            rct_temp.append(int(rct[i][j][1:]))
            
        for j in range(len(prd[i])):
            prd_temp.append(int(prd[i][j][1:]))
        
        if len(mod_type[i]) == 0:
            regType = 0
        else:
            for k in range(len(mod_type[i])):
                if mod_type[i][k] == 'inhibitor':
                    inhib.append(int(mod[i][k][1:]))
                elif mod_type[i][k] == 'activator':
                    activ.append(int(mod[i][k][1:]))
                
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
    
    r = te.loada(antStr)
    
    numBnd = r.getNumBoundarySpecies()
    numFlt = r.getNumFloatingSpecies()
    boundaryId = r.getBoundarySpeciesIds()
    floatingId = r.getFloatingSpeciesIds()
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
    prd = []
    mod = []
    r_type = []
    kineticLaw = []
    mod_type = []
    
    doc = tesbml.readSBMLFromString(r.getSBML())
    sbmlmodel = doc.getModel()

    for slr in sbmlmodel.getListOfReactions():
        temprct = []
        tempprd = []
        tempmod = []
        
        sbmlreaction = sbmlmodel.getReaction(slr.getId())
        for sr in range(sbmlreaction.getNumReactants()):
            sbmlrct = sbmlreaction.getReactant(sr)
            temprct.append(sbmlrct.getSpecies())
        for sp in range(sbmlreaction.getNumProducts()):
            sbmlprd = sbmlreaction.getProduct(sp)
            tempprd.append(sbmlprd.getSpecies())
        for sm in range(sbmlreaction.getNumModifiers()):
            sbmlmod = sbmlreaction.getModifier(sm)
            tempmod.append(sbmlmod.getSpecies())
        kl = sbmlreaction.getKineticLaw()
        
        rct.append(sorted(temprct, key=lambda v: (v.upper(), v[0].islower())))
        prd.append(sorted(tempprd, key=lambda v: (v.upper(), v[0].islower())))
        mod.append(sorted(tempmod, key=lambda v: (v.upper(), v[0].islower())))
        
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
            if n.has(mod[ml][ml_i]) and not d.has(mod[ml][ml_i]):
                mod_type_temp.append('activator')
            elif d.has(mod[ml][ml_i]) and not n.has(mod[ml][ml_i]):
                mod_type_temp.append('inhibitor')
            elif n.has(mod[ml][ml_i]) and d.has(mod[ml][ml_i]):
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
        elif len(rct[i]) == 2:
            if len(prd[i]) == 1:
                rType = 1
            elif len(prd[i]) == 2:
                rType = 3
        
        for j in range(len(rct[i])):
            rct_temp.append(int(rct[i][j][1:]))
            
        for j in range(len(prd[i])):
            prd_temp.append(int(prd[i][j][1:]))
        
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


def generateSimpleRateLaw(rl, floatingIds, boundaryIds, Jind):
    
    Klist = []
    
    T = ''
    D = ''
    ACT = ''
    INH = ''
    
    # T
    T = T + '(Kf' + str(Jind) + '*'
    Klist.append('Kf' + str(Jind))
    
    for i in range(len(rl[Jind][3])):
        T = T + 'S' + str(rl[Jind][3][i])
        if i < len(rl[Jind][3]) - 1:
            T = T + '*'
    
    if rl[Jind][2] == Reversibility.REVERSIBLE:
        T = T + ' - Kr' + str(Jind) + '*'
        Klist.append('Kr' + str(Jind))
        
        for i in range(len(rl[Jind][4])):
            T = T + 'S' + str(rl[Jind][4][i])
            if i < len(rl[Jind][4]) - 1:
                T = T + '*'
            
    T = T + ')'
        
    # D
    D = D + '1 + '
    
    for i in range(len(rl[Jind][3])):
        D = D + 'S' + str(rl[Jind][3][i])
        if i < len(rl[Jind][3]) - 1:
            D = D + '*'
    
    if rl[Jind][2] == Reversibility.REVERSIBLE:
        D = D + ' + '
        for i in range(len(rl[Jind][4])):
            D = D + 'S' + str(rl[Jind][4][i])
            if i < len(rl[Jind][4]) - 1:
                D = D + '*'
    
    # Activation
    if (len(rl[Jind][5]) > 0):
        for i in range(len(rl[Jind][5])):
            ACT = ACT + '(1 + Ka' + str(Jind) + str(i) + '*'
            Klist.append('Ka' + str(Jind) + str(i))
            ACT = ACT + 'S' + str(rl[Jind][5][i]) + ')*'
            
    # Inhibition
    if (len(rl[Jind][6]) > 0):
        for i in range(len(rl[Jind][6])):
            INH = INH + '(1/(1 + Ki' + str(Jind) + str(i) + '*'
            Klist.append('Ki' + str(Jind) + str(i))
            INH = INH + 'S' + str(rl[Jind][6][i]) + '))*'
    
    rateLaw = ACT + INH + T + '/(' + D + ')'
        
    return rateLaw, Klist


def generateAntimony(floatingIds, boundaryIds, stt1, stt2, reactionList, boundary_init=None):
    reactionListCopy = copy.deepcopy(reactionList)
    Klist = []
    
    real = np.append(floatingIds, boundaryIds)
    if type(real[0]) == 'str' or type(real[0]) == np.str_:
        real = [s.strip('S') for s in real]
    real = list(map(int, real))
    tar = stt1 + stt2
    tar = list(map(int, tar))
    
    # List species
    antStr = ''
    if len (floatingIds) > 0:
        antStr = antStr + 'var ' + str(floatingIds[0])
        for index in floatingIds[1:]:
            antStr = antStr + ', ' + str(index)
        antStr = antStr + ';\n'
    
    if len (boundaryIds) > 0:
        antStr = antStr + 'const ' + str(boundaryIds[0])
        for index in boundaryIds[1:]:
            antStr = antStr + ', ' + str(index)
        antStr = antStr + ';\n'

    # List reactions
    for index, rind in enumerate(reactionListCopy):
        if rind[0] == ReactionType.UNIUNI:
            # UniUni
            antStr = antStr + 'J' + str(index) + ': S' + str(real[tar.index(reactionListCopy[index][3][0])])
            antStr = antStr + ' -> '
            antStr = antStr + 'S' + str(real[tar.index(reactionListCopy[index][4][0])])
            antStr = antStr + '; '
            RateLaw, klist_i = generateSimpleRateLaw(reactionList, floatingIds, boundaryIds, index)
            antStr = antStr + RateLaw
            Klist.append(klist_i)
        elif rind[0] == ReactionType.BIUNI:
            # BiUni
            antStr = antStr + 'J' + str(index) + ': S' + str(real[tar.index(reactionListCopy[index][3][0])])
            antStr = antStr + ' + '
            antStr = antStr + 'S' + str(real[tar.index(reactionListCopy[index][3][1])])
            antStr = antStr + ' -> '
            antStr = antStr + 'S' + str(real[tar.index(reactionListCopy[index][4][0])])
            antStr = antStr + '; '
            RateLaw, klist_i = generateSimpleRateLaw(reactionList, floatingIds, boundaryIds, index)
            antStr = antStr + RateLaw
            Klist.append(klist_i)
        elif rind[0] == ReactionType.UNIBI:
            # UniBi
            antStr = antStr + 'J' + str(index) + ': S' + str(real[tar.index(reactionListCopy[index][3][0])])
            antStr = antStr + ' -> '
            antStr = antStr + 'S' + str(real[tar.index(reactionListCopy[index][4][0])])
            antStr = antStr + ' + '
            antStr = antStr + 'S' + str(real[tar.index(reactionListCopy[index][4][1])])
            antStr = antStr + '; '
            RateLaw, klist_i = generateSimpleRateLaw(reactionList, floatingIds, boundaryIds, index)
            antStr = antStr + RateLaw
            Klist.append(klist_i)
        else:
            # BiBi
            antStr = antStr + 'J' + str(index) + ': S' + str(real[tar.index(reactionListCopy[index][3][0])])
            antStr = antStr + ' + '
            antStr = antStr + 'S' + str(real[tar.index(reactionListCopy[index][3][1])])
            antStr = antStr + ' -> '
            antStr = antStr + 'S' + str(real[tar.index(reactionListCopy[index][4][0])])
            antStr = antStr + ' + '
            antStr = antStr + 'S' + str(real[tar.index(reactionListCopy[index][4][1])])
            antStr = antStr + '; '
            RateLaw, klist_i = generateSimpleRateLaw(reactionList, floatingIds, boundaryIds, index)
            antStr = antStr + RateLaw
            Klist.append(klist_i)
        antStr = antStr + ';\n'

    # List rate constants
    antStr = antStr + '\n'
    Klist_f = [item for sublist in Klist for item in sublist]
    
    for i in range(len(Klist_f)):
        if Klist_f[i].startswith('Kf'):
            antStr = antStr + Klist_f[i] + ' = 1\n'
        elif Klist_f[i].startswith('Kr'):
            antStr = antStr + Klist_f[i] + ' = 0.5\n'
        elif Klist_f[i].startswith('Ka'):
            antStr = antStr + Klist_f[i] + ' = 1\n'
        elif Klist_f[i].startswith('Ki'):
            antStr = antStr + Klist_f[i] + ' = 1\n'
        
    # Initialize boundary species
    antStr = antStr + '\n'
    if type(boundary_init) == type(None):
        for index, bind in enumerate(boundaryIds):
            antStr = antStr + str(bind) + ' = ' + str(np.random.randint (1,6)) + '\n'
    else:
        for index, bind in enumerate(boundaryIds):
            antStr = antStr + str(bind) + ' = ' + str(boundary_init[index]) + '\n'
    
    # Initialize floating species
    for index, find in enumerate(floatingIds):
        antStr = antStr + str(find) + ' = ' + '1\n'
        
    return antStr
     

def generateParameterBoundary(glgp):
    
    pBound = []
    
    for i in range(len(glgp)):
        if glgp[i].startswith('Kf'):
            pBound.append((1e-3, 100.))
        elif glgp[i].startswith('Kr'):
            pBound.append((1e-3, 100.))
        elif glgp[i].startswith('Ka'):
            pBound.append((1e-3, 100.))
        elif glgp[i].startswith('Ki'):
            pBound.append((1e-3, 100.))

    return pBound
    

    
