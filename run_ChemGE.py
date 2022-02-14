import numpy as np
import pandas as pd
import json
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools,MolFromSmiles
from rdkit.ML.Descriptors import MoleculeDescriptors
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
import sys
import os
sys.path.append("./ChemGE")
from ChemGE import optimize_blox

with open("config.json", "r") as f:
        config = json.load(f)


def build_model(prediction_model, x_train, y_train):
    if prediction_model == 'RF':
        params = {'n_estimators':[10, 50, 100]}
        gridsearch = GridSearchCV(RandomForestRegressor(), param_grid=params, cv = 3, scoring="r2", n_jobs=config["parallel"], verbose = 0)
        gridsearch.fit(x_train,y_train)
        model =  RandomForestRegressor(n_estimators = gridsearch.best_params_['n_estimators'])
        model.fit(x_train, y_train)
        return model



def recommend_next(prediction_model, features_observed, features_unchecked, properties_observed):
    sc_property = StandardScaler()
    sc_property.fit(properties_observed)
    sc_properties_observed = sc_property.transform(properties_observed)

    model_list = []
    for d in range(2):
        model = build_model(prediction_model, features_observed, properties_observed[:, d])
        model_list.append(model)

    best_smi, SN = optimize_blox.run_optimize(sc_properties_observed, model_list, sc_property)
    m = MolFromSmiles(best_smi)
    maccskey = [x for x in AllChem.GetMACCSKeysFingerprint(m)]
    predicted_properties = []
    for d in range(2):
        predicted_properties.append(model_list[d].predict([maccskey])[0])

    return m, predicted_properties, SN ,maccskey

def main():
    df = pd.read_csv(config["data_dir"], names=['smiles'])
    PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')

    none_list = []
    for i in range(df.shape[0]):
        if Chem.MolFromSmiles(df['smiles'][i]) is None:
            none_list.append(i)
            
    df = df.drop(none_list)
    mols = [Chem.MolFromSmiles(smile) for smile in df['smiles']]

    maccskeys = []
    for m in mols:
        maccskey = [x for x in AllChem.GetMACCSKeysFingerprint(m)]
        maccskeys.append(maccskey)

    descriptor_names = ['qed', 'MolLogP']
    descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
    descriptors = pd.DataFrame(
        [descriptor_calculator.CalcDescriptors(mol) for mol in mols[:1000]],
        columns = descriptor_names
    )
    properties = np.array(descriptors)
    data = np.array(maccskeys[:1000])

    features_observed = data[:10]
    features_unchecked = data[10:]

    properties_observed = properties[:10]
    properties_unchecked = properties[10:]

    smiles_observed = df.smiles[:10].to_numpy()
    num_loop = config["loop_num"]

    l = 0
    while l < num_loop:
        print('Exploration:', l)
        m, predicted_properties, SN, maccskey = recommend_next('RF', features_observed, features_unchecked,
                                                            properties_observed)
        # Add the experimental or simulation result of the recommended data
        new_smiles = Chem.MolToSmiles(m)
        if new_smiles in smiles_observed:
            continue
        features_observed = np.append(features_observed, [maccskey], axis=0)
        new_properties_abserved = descriptor_calculator.CalcDescriptors(m)
        properties_observed = np.append(properties_observed, [new_properties_abserved], axis = 0)
        smiles_observed = np.append(smiles_observed, [new_smiles], axis = 0)
        l += 1

    if not os.path.exists("./fig"):
        os.mkdir("./fig")
    if not os.path.exists("./results"):
        os.mkdir("./results")

    plt.scatter(properties_observed[:10, 0], properties_observed[:10, 1], label='Initial data')
    plt.scatter(properties_observed[10:, 0], properties_observed[10:, 1], label='Experimental data')
    plt.title(f"n={config['loop_num']} generation={config['generation_num']}")
    plt.xlabel('qed')
    plt.ylabel('MolLogP')
    plt.xlim((0, 1))
    plt.ylim((-8, 12))
    plt.legend()
    plt.savefig(f"./fig/{config['evolution_mode']}_n{config['loop_num']}_g{config['generation_num']}.png", dpi=300)
    plt.close()

    df_properties_generated = pd.DataFrame(properties_observed[10:])
    df_properties_generated.to_csv(f"./results/df_{config['evolution_mode']}_properties_generated.csv", header=False, index=0)

if __name__ == "__main__":
    main()