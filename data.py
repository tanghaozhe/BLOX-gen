import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, Draw, Descriptors, PandasTools,QED
import matplotlib.pyplot as plt



df = pd.read_csv('ZINC_first_1000.smi',names=['smiles'])
PandasTools.AddMoleculeColumnToFrame(frame=df, smilesCol='smiles')
df['QED'] = df.ROMol.map(QED.qed)
df['MolLogP'] = df.ROMol.map(Descriptors.MolLogP)

def fix_data(data):
    x = data[0]
    y = data[1]
    p = np.arctanh(x)
    y = y - 3*p
    return pd.Series([x, y])

df[['QED_fixed','MolLogP_fixed']] = df[['QED','MolLogP']].apply(fix_data,axis=1)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('raw data')
plt.xlim((0, 1))
plt.ylim((-6, 8))
plt.xlabel('QED')
plt.ylabel('MolLogP')
plt.scatter(df['QED'],df['MolLogP'])

plt.subplot(1, 2, 2)
plt.title('fixed data')
plt.xlim((0, 1))
plt.ylim((-6, 8))
plt.xlabel('QED')
plt.ylabel('MolLogP')
plt.scatter(df['QED_fixed'],df['MolLogP_fixed'])

plt.show()

#               QED     MolLogP
# count  999.000000  999.000000
# mean     0.728276    2.542245
# std      0.139405    1.472247
# min      0.155610   -2.946800
# 25%      0.645571    1.702410
# 50%      0.753442    2.681000
# 75%      0.836122    3.609170
# max      0.945932    6.152520