import pandas as pd
import deepchem as dc
import os

from tdc.single_pred import ADME
from tdc.single_pred import Tox

data = ADME(name = 'BBB_Martins')
dataset_name = 'BBBM'
hyper_opt_dataset = os.path.join(dataset_name, dataset_name + '_hyper_opt.csv')
training_dataset = os.path.join(dataset_name, dataset_name + '_training.csv')

data.get_data(format='df').to_csv('data_.csv')

featurizer = dc.feat.CircularFingerprint(size=1024)
loader = dc.data.CSVLoader(tasks=['Y'], feature_field="Drug",featurizer=featurizer)
dataset = loader.featurize('data_.csv')
splitter = dc.splits.ScaffoldSplitter()
scaffold_list = splitter.split(dataset, frac_train = 1/3, frac_test = 2/3, frac_valid = 0)

hyper_opt_df = pd.DataFrame({"smiles": data.get_data(format='df')['Drug'][scaffold_list[0]]})
hyper_opt_df['label'] = dataset.y[scaffold_list[0]]
hyper_opt_df.to_csv(hyper_opt_dataset, index=False)

training_df = pd.DataFrame({"smiles": data.get_data(format='df')['Drug'][scaffold_list[2]]})
training_df['label'] = dataset.y[scaffold_list[2]]
training_df.to_csv(training_dataset, index=False)