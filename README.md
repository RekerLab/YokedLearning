# YokedLearning
This repository shows an example pipeline to evaluate the yoked learning performance on both classical machine learning (part 1) and deep learning model (part 2)

## Files 
- **Part1 Yoked leraning** contains all codes and functions to run and evaluate yoked learning with classical machine learning models (random forest, naive bayes and logistic regression)
    - **yoked_machine_learning_pipeline.py** contains the functions for yoked learning 
    - **yoked_machine_leraning_pipeline.py** contains the main function to run the yoked learning 
    - **example boxplot/lineplot.ipynb** contains an example notebook that visualize the comparison among yoked learning, active learning and passive learning

- **Part2 Yoked MLP** contains all codes and functions to evaluate yoked leanring with yoked deep learning models. The yoked deep learning pipeline will be ran through ActiveLearningBenchmark repository: https://github.com/RekerLab/ActiveLearningBenchmark. You could run the yoked deep learning through default MLP parameters (ffn_num_layers: 2, ffn_hidden_size: 300, dropout: 0, batch_size:50), or through optimized MLP parameters (We obtained our optimized MLP parameters through chemprop repository: https://github.com/chemprop/chemprop, but you are welcomed to use your own algorithm)
    - **Single_dataset_comparison.ipynb** contains an example notebook that visualzie the comparison for the outputs

For example, this is how we run the yoked deep learnig: 
```commandline
python3 ActiveLearning.py --data_path alb/data/bace.csv --pure_columns mol --target_columns Class --dataset_type classification --metrics roc-auc mcc accuracy precision recall f1_score --learning_type explorative --model_config_selector model_config/RandomForest_Morgan_Config --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 10 --seed 0 --save_dir test_al --model_config_extra_evaluators model_config/MLP_BinaryClassification_Morgan_Config
```

## Dependencies
This code uses numpy, scipy, sklearn, numpy, deepchem, seaborn, matplotlib, chemprop, Therapeutics Data Commons (https://github.com/mims-harvard/TDC), and ActiveLearningBenchmark (https://github.com/RekerLab/ActiveLearningBenchmark).