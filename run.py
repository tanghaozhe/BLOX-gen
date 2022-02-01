import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, PandasTools
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('ZINC_first_1000.smi',names=['smiles'])
PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')

none_list=[]
for i in range(df.shape[0]):
    if Chem.MolFromSmiles(df['smiles'][i]) is None:
        none_list.append(i)
        
df=df.drop(none_list)
mols=[Chem.MolFromSmiles(smile) for smile in df['smiles']]

maccskeys = []
for m in mols:
    maccskey = [x for x in AllChem.GetMACCSKeysFingerprint(m)]
    maccskeys.append(maccskey)

descriptor_names = ['qed', 'MolLogP']
descriptor_calculator = MoleculeDescriptors.MolecularDescriptorCalculator(descriptor_names)
descriptors = pd.DataFrame(
    [descriptor_calculator.CalcDescriptors(mol) for mol in mols[:1000]],
    columns=descriptor_names
)
properties=np.array(descriptors)

# fix data
def foo(data):
    x = data[0]
    y = data[1]
    y = y - 3 * np.arctanh(x)
    return [x,y]

properties = np.array(list(map(foo, properties)))

data=np.array(maccskeys[:1000])

features_observed = data[:10]
features_unchecked = data[10:]

properties_observed = properties[:10]
properties_unchecked = properties[10:]


def build_model(prediction_model, x_train, y_train):
    if prediction_model == 'RF':
        params = {'n_estimators':[10, 50, 100]}
        gridsearch = GridSearchCV(RandomForestRegressor(), param_grid=params, cv = 3, scoring="r2", n_jobs=parallel, verbose = 0)
        gridsearch.fit(x_train,y_train)
        model =  RandomForestRegressor(n_estimators = gridsearch.best_params_['n_estimators'])
        model.fit(x_train, y_train)
        return model

parallel = 1


def hesgau(x, y, sigma):
    dim = len(x)
    dist = np.sum(np.power(x-y, 2))
    return (dim/sigma - dist/sigma**2)*np.exp(-dist/(2*sigma))

def stein_novelty(point, data_list, sigma):
    n = len(data_list)
    score = 0
    score = np.sum([hesgau(point, data_list[k,:], sigma) for k in range(n)])
    score = score/(n*(n+1)/2)
    return -score

# sc_predicted_properties_list = sc_property.transform(predicted_properties_list)
# stein_novelty(sc_predicted_properties_list[0], sc_properties_observed, sigma=1)

def recommend_next(prediction_model, features_observed, features_unchecked, properties_observed):
    sc = StandardScaler()
    sc.fit(features_observed)
    sc_features_observed = sc.transform(features_observed)
    sc_features_unchecked = sc.transform(features_unchecked)
    sc_property = StandardScaler()
    sc_property.fit(properties_observed)
    sc_properties_observed = sc_property.transform(properties_observed)

    model_list = []
    for d in range(2):
        model = build_model(prediction_model, sc_features_observed, properties_observed[:, d])
        model_list.append(model)

    predicted_properties_list = []
    for d in range(2):
        predicted_properties_list.append(model_list[d].predict(sc_features_unchecked))

    predicted_properties_list = np.array(predicted_properties_list).T

    # Calc. Stein Novelty
    sc_predicted_properties_list = sc_property.transform(predicted_properties_list)
    sn_data = [stein_novelty(point, sc_properties_observed, sigma=1) for point in sc_predicted_properties_list]

    # Select and save next candidate
    maximum_index = np.argmax(sn_data)

    return maximum_index, predicted_properties_list[maximum_index], sn_data[maximum_index]

num_loop=100

for l in range(num_loop):
        print('Exploration:', l)
        recommended_index, predicted_properties, SN = recommend_next('RF', features_observed, features_unchecked, properties_observed)
        print('Recommended_index', recommended_index, 'predicted_properties', predicted_properties, 'Stein novelty', SN)

        #Add the experimental or simulation result of the recommended data
        features_observed = np.append(features_observed, [features_unchecked[recommended_index]], axis = 0)
        properties_observed = np.append(properties_observed, [properties_unchecked[recommended_index]], axis = 0)

        #Removed the recommend data
        features_unchecked = np.delete(features_unchecked, recommended_index, axis = 0)
        properties_unchecked = np.delete(properties_unchecked, recommended_index, axis = 0)

        # plt.scatter(properties_observed[:-1, 0], properties_observed[:-1, 1], label='Prev data')
        # plt.scatter([predicted_properties[0]], [predicted_properties[1]], label='Predicted properties')
        # plt.scatter(properties_observed[-1:, 0], properties_observed[-1:, 1], label='Experimental data')
        # plt.xlim((0, 1))
        # plt.ylim((-6, 8))
        # plt.xlabel('QED')
        # plt.ylabel('MolLogP')
        # plt.legend()
        # plt.savefig('fig/observed_data_iteration' + str(l) + '.png', dpi=300)
        # plt.close()
plt.figure(figsize=(6, 6))
plt.scatter(properties_observed[:, 0], properties_observed[:, 1])
plt.xlim((0, 1))
plt.ylim((-6, 8))
plt.xlabel('QED')
plt.ylabel('MolLogP')
plt.savefig('fig_1000.png', dpi=300)
plt.close()