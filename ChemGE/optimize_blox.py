from __future__ import print_function
import copy
import nltk
import numpy as np
from rdkit import Chem
from rdkit import rdBase
from ChemGE import zinc_grammar,cfg_util,score_util
import json
rdBase.DisableLog('rdApp.error')
GCFG = zinc_grammar.GCFG

with open("config.json", "r") as f:
    config = json.load(f)

def CFGtoGene(prod_rules, max_len=-1):
    gene = []
    for r in prod_rules:
        lhs = GCFG.productions()[r].lhs()
        possible_rules = [idx for idx, rule in enumerate(GCFG.productions())
                          if rule.lhs() == lhs]
        gene.append(possible_rules.index(r))
    if max_len > 0:
        if len(gene) > max_len:
            gene = gene[:max_len]
        else:
            gene = gene + [np.random.randint(0, 256)
                           for _ in range(max_len-len(gene))]
    return gene


def GenetoCFG(gene):
    prod_rules = []
    stack = [GCFG.productions()[0].lhs()]
    for g in gene:
        try:
            lhs = stack.pop()
        except Exception:
            break
        possible_rules = [idx for idx, rule in enumerate(GCFG.productions())
                          if rule.lhs() == lhs]
        rule = possible_rules[g % len(possible_rules)]
        prod_rules.append(rule)
        rhs = filter(lambda a: (type(a) == nltk.grammar.Nonterminal)
                     and (str(a) != 'None'),
                     zinc_grammar.GCFG.productions()[rule].rhs())
        stack.extend(list(rhs)[::-1])
    return prod_rules


def mutation(gene):
    idx = np.random.choice(len(gene))
    gene_mutant = copy.deepcopy(gene)
    gene_mutant[idx] = np.random.randint(0, 256)
    return gene_mutant


def canonicalize(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if smiles != '' and mol is not None and mol.GetNumAtoms() > 1:
        return Chem.MolToSmiles(mol)
    else:
        return smiles


elapsed_min = 0
best_score = 0
mean_score = 0
std_score = 0
min_score = 0
best_smiles = ""
all_smiles = []



def run_optimize(sc_properties_observed, model_list, sc_property):
    global best_smiles
    global best_score
    global all_smiles

    gene_length = 300

    N_mu = 100
    N_lambda = 200

    # initialize population
    seed_smiles = []
    with open(config["data_dir"]) as f:
        for line in f:
            smiles = line.rstrip()
            seed_smiles.append(smiles)

    initial_smiles = np.random.choice(seed_smiles, N_mu+N_lambda)
    initial_smiles = [canonicalize(s) for s in initial_smiles]
    initial_genes = [CFGtoGene(cfg_util.encode(s), max_len=gene_length)
                     for s in initial_smiles]
    initial_scores = [score_util.calc_score(s,sc_properties_observed, model_list, sc_property) for s in initial_smiles]

    population = []
    for score, gene, smiles in zip(initial_scores, initial_genes,
                                   initial_smiles):
        population.append((score, smiles, gene))

    population = sorted(population, key=lambda x: x[0], reverse=True)[:N_mu]

    all_smiles = [p[1] for p in population]
    for generation in range(config["generation_num"]):
        scores = [p[0] for p in population]
        mean_score = np.mean(scores)
        min_score = np.min(scores)
        std_score = np.std(scores)
        best_score = np.max(scores)
        idx = np.argmax(scores)
        best_smiles = population[idx][1]
        print("% generation:{},best_score:{:.5f},mean_score:{:.2f},mean_score:{:.2f},std_score:{:.2f}".format(generation, best_score,
                                       mean_score, min_score, std_score))

        new_population = []
        for _ in range(N_lambda):
            p = population[np.random.randint(len(population))]
            p_gene = p[2]
            c_gene = mutation(p_gene)

            c_smiles = canonicalize(cfg_util.decode(GenetoCFG(c_gene)))
            if c_smiles not in all_smiles:
                c_score = score_util.calc_score(c_smiles,sc_properties_observed,model_list, sc_property)
                c = (c_score, c_smiles, c_gene)
                new_population.append(c)
                all_smiles.append(c_smiles)

        population.extend(new_population)
        population = sorted(population,
                            key=lambda x: x[0], reverse=True)[:N_mu]
    return population[0][1], population[0][0]
