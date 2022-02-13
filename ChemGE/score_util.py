import networkx as nx
from rdkit.Chem import AllChem as Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MolFromSmiles
from rdkit.Chem import rdmolops
from rdkit import rdBase
import numpy as np
import sascorer
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools
rdBase.DisableLog('rdApp.error')


# from https://github.com/gablg1/ORGAN/blob/master/organ/mol_metrics.py#L83
def verify_sequence(smile):
    mol = Chem.MolFromSmiles(smile)
    return smile != '' and mol is not None and mol.GetNumAtoms() > 1

# from grammar VAE
# logP_values = np.loadtxt('logP_values.txt')
# SA_scores = np.loadtxt('SA_scores.txt')
# cycle_scores = np.loadtxt('cycle_scores.txt')
logP_mean = 2.457    # np.mean(logP_values)
logP_std = 1.434     # np.std(logP_values)
SA_mean = -3.053     # np.mean(SA_scores)
SA_std = 0.834       # np.std(SA_scores)
cycle_mean = -0.048  # np.mean(cycle_scores)
cycle_std = 0.287    # np.std(cycle_scores)

def hesgau(x, y, sigma):
    dim = len(x)
    dist = np.sum(np.power(x-y, 2))
    return (dim/sigma - dist/sigma**2)*np.exp(-dist/(2*sigma))

def stein_novelty(point, data_list, sigma):
    n = len(data_list)
    score = np.sum([hesgau(point, data_list[k,:], sigma) for k in range(n)])
    score = score/(n*(n+1)/2)
    return -score

def calc_score(smiles,sc_properties_observed,model_list, sc_property):
    if verify_sequence(smiles):
        m = MolFromSmiles(smiles)
        maccskey = [x for x in AllChem.GetMACCSKeysFingerprint(m)]
        predicted_properties = []
        for d in range(2):
            predicted_properties.append(model_list[d].predict([maccskey])[0])
        sc_predicted_properties = sc_property.transform([predicted_properties])[0]
        sn = stein_novelty(sc_predicted_properties, sc_properties_observed, sigma=1)
        # return sn
        return 1
    else:
        return -1e10


# def calc_score(smiles,sc_properties_observed):
#     if verify_sequence(smiles):
#         try:
#             molecule = MolFromSmiles(smiles)
#             if Descriptors.MolWt(molecule) > 500:
#                 return -1e10
#             current_log_P_value = Descriptors.MolLogP(molecule)
#             current_SA_score = -sascorer.calculateScore(molecule)
#             cycle_list = nx.cycle_basis(nx.Graph(rdmolops.GetAdjacencyMatrix(molecule)))
#             if len(cycle_list) == 0:
#                 cycle_length = 0
#             else:
#                 cycle_length = max([len(j) for j in cycle_list])
#             if cycle_length <= 6:
#                 cycle_length = 0
#             else:
#                 cycle_length = cycle_length - 6
#             current_cycle_score = -cycle_length
#
#             current_SA_score_normalized = (current_SA_score - SA_mean) / SA_std
#             current_log_P_value_normalized = (current_log_P_value - logP_mean) / logP_std
#             current_cycle_score_normalized = (current_cycle_score - cycle_mean) / cycle_std
#
#             score = (current_SA_score_normalized
#                      + current_log_P_value_normalized
#                      + current_cycle_score_normalized)
#             return score
#         except Exception:
#             return -1e10
#     else:
#         return -1e10