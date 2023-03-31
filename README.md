# YokedLearning

![YokedLearning2](https://user-images.githubusercontent.com/127516906/229117268-a5e11bc9-1414-4bad-8cff-a88e833eccfd.png)


Yoked machine learning utilizes a teacher model to guide a student model. We provide an example pipeline to evaluate yoked learning performance on both classical (part 1) and deep (part 2) machine learning models.

## Benchmarking Datasets
* [Therapeutics Data Commons](https://github.com/mims-harvard/TDC)
* [DeepChem](https://deepchem.io)
* [MoleculeNet](https://moleculenet.org)

## Dependencies
* [numpy](https://numpy.org/)
* [scipy](https://scipy.org/)
* [pandas](https://github.com/pandas-dev/pandas)
* [scikit-learn](https://scikit-learn.org/stable/)
* [DeepChem](https://deepchem.io/)
* [matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)
* [chemprop v1.5.2](https://github.com/chemprop/chemprop)
* [ActiveLearningBenchmark](https://github.com/RekerLab/ActiveLearningBenchmark).

## Files 
#### Part 1: Classical Yoked Learning
- Code and functions to evaluate yoked learning with classical machine learning models (random forest, naive bayes and logistic regression).
    - **yoked_machine_learning_pipeline.py** contains functions for evaluating yoked learning 
    - **yoked_learning_main.py** contains the main function to run yoked learning 
    - **example boxplot/lineplot.ipynb** contains an example notebook that visualize comparisons between yoked learning, active learning, and passive learning

#### Part 2: Deep Yoked Learning
- Code and functions to evaluate yoked learning with deep learning models (MLP). 
  - Implementation methods include [ActiveLearningBenchmark repository](https://github.com/RekerLab/ActiveLearningBenchmark), default MLP parameters (ffn_num_layers: 2, ffn_hidden_size: 300, dropout: 0, batch_size:50) or optimized MLP parameters [chemprop](https://github.com/chemprop/chemprop)
  - **Single_dataset_comparison.ipynb** contains an example notebook that visualizes the output comparisons
 - Example implementation: 
```commandline
python3 ActiveLearning.py --data_path alb/data/bace.csv --pure_columns mol --target_columns Class --dataset_type classification 
--metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative 
--model_config_selector model_config/RandomForest_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 
--evaluate_stride 10 --seed 0 --save_dir test_al --model_config_extra_evaluators model_config/MLP_BinaryClassification_Morgan_Config
```
