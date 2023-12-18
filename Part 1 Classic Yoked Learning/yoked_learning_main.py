# Import necessary libraries
import sklearn
import pandas as pd
import numpy as np
import deepchem as dc

import pickle
import multiprocessing as mp
from multiprocessing import Pool

from random import randrange
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

from tdc import Evaluator
from tdc.single_pred import ADME
from tdc.single_pred import Tox
from sklearn.metrics import matthews_corrcoef as mcc

from yoked_machine_learning_pipeline import teach

# Define Models
model_rf = RF(n_jobs = -1)
model_log = LogisticRegression(n_jobs = -1)
model_nb = BernoulliNB()

# Define students
students_list = []
students_list.append(('RF', model_rf))
students_list.append(('LOG', model_log))
students_list.append(('NB', model_nb))

# Initialize Metrics:
metrics_list = []
metrics_list.append(('mccs_auc', mcc))
metrics_name = [list_name[0] for list_name in metrics_list]

# Teachers initialization
teachers_names = ['RF', 'LOG', 'NB', 'PASSIVE']
teacher_list = []
teacher_list.append(model_rf)
teacher_list.append(model_log)
teacher_list.append(model_nb)
teacher_list.append(None)
    
if __name__ == '__main__':
    
    # Datasets come from https://tdcommons.ai
    data = ADME(name = 'HIA_Hou')
    dataset_name = 'HIA'
    
    output_file = dataset_name + '_results'
    final_result = {}
    teacher_results = []
    repeats = 1
    
    # choose featurizer here by switching featurizer definition
    featurizer = dc.feat.CircularFingerprint()
    # featuzier = dc.feat.MACCSKeysFingerprint()
    # featurizer = dc.feat.RDKitDescriptors()
    
    # Multiprocess running of yoked learning
    with Pool(processes=5) as pool:
        workers = [pool.apply_async(teach, args=(teacher, students_list, data, featurizer, metrics_list, repeats, False)) 
                   for teacher in teacher_list]
        teacher_results = [worker.get() for worker in workers]   
        
    # Convert the output into a dictionary labeled with the teacher and student names as keys
    index = 0
    for name in teachers_names:
        final_result[name] = teacher_results[index]
        index = index + 1

    # Save the results as a file
    outfile = open(output_file,'wb')
    pickle.dump(final_result,outfile)
    outfile.close()

