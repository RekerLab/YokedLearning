

![YokedLearning4](https://user-images.githubusercontent.com/127516906/229135399-2e586506-45b3-4731-8192-6c356c666963.png)



Yoked machine learning utilizes a teacher model to guide a student model. We provide an example pipeline to evaluate yoked learning performance on both classical (part 1) and deep (part 2) machine learning models.

## Benchmarking Datasets
* [Therapeutics Data Commons](https://github.com/mims-harvard/TDC)
  * ADME: Pharmaco-kinetics (from tdc.single_pred import ADME)
      * [CYP2C9 Substrate](https://tdcommons.ai/single_pred_tasks/adme/), Carbon-Mangels et al.
        * data = ADME(name = 'CYP2C9_Substrate_CarbonMangels')
      * [CYP2D6 Substrate](https://tdcommons.ai/single_pred_tasks/adme/), Carbon-Mangels et al.
        * data = ADME(name = 'CYP2D6_Substrate_CarbonMangels')
      * [CYP3A4 Substrate](https://tdcommons.ai/single_pred_tasks/adme/), Carbon-Mangels et al. 
        * data = ADME(name = 'CYP3A4_Substrate_CarbonMangels')
      * [HIA](https://tdcommons.ai/single_pred_tasks/adme/) (Human Intestinal Absorption), Hou et al.
        * data = ADME(name = 'HIA_Hou')
      * [Pgp](https://tdcommons.ai/single_pred_tasks/adme/) (P-glycoprotein) Inhibition, Broccatelli et al.
        * data = ADME(name = 'Pgp_Broccatelli')
      * [Bioavailability](https://tdcommons.ai/single_pred_tasks/adme/), Ma et al.
        * data = ADME(name = 'Bioavailability_Ma')
       
  * Tox: Toxicity (from tdc.single_pred import Tox)
      * [hERG blockers](https://tdcommons.ai/single_pred_tasks/tox/), Wang et al.
        * data = Tox(name = 'hERG')
      * [DILI](https://tdcommons.ai/single_pred_tasks/tox/) (Drug Induced Liver Injury), Xu et al.
        * data = Tox(name = 'DILI')
      * [Skin Reaction](https://tdcommons.ai/single_pred_tasks/tox/), Alves et al.
        * data = Tox(name = 'Skin Reaction')
      * [Carcinogens](https://tdcommons.ai/single_pred_tasks/tox/), Lagunin, et al.
        * data = Tox(name = 'Carcinogens_Lagunin')
      * [Clintox](https://tdcommons.ai/single_pred_tasks/tox/), Gayvert, et al.
        * data = Tox(name = 'ClinTox')
  
  * HTS: High-Throughput Screening (from tdc.single_pred import HTS)
      * [SARS-CoV-2 3CL Protease](https://tdcommons.ai/single_pred_tasks/hts/), Diamond
        * data = HTS(name = 'SARSCoV2_3CLPro_Diamond')
          
* [MoleculeNet](https://moleculenet.org)
    * [BACE](https://moleculenet.org/datasets-1): Quantitative (IC50) and qualitative (binary label) binding results for a set of inhibitors of human β-secretase 1(BACE-1)
    * [BBBP](https://moleculenet.org/datasets-1): Binary labels of blood-brain barrier penetration(permeability)

## Dependencies
* [numpy](https://numpy.org/)
* [scipy](https://scipy.org/)
* [pandas](https://github.com/pandas-dev/pandas)
* [scikit-learn](https://scikit-learn.org/stable/)
* [DeepChem](https://deepchem.io/)
* [matplotlib](https://matplotlib.org/)
* [seaborn](https://seaborn.pydata.org/)
* [chemprop v1.5.2](https://github.com/chemprop/chemprop)
* [MolALKit](https://github.com/RekerLab/MolALKit)

## Files 
#### Part 1: Classical Yoked Learning
- Code and functions to evaluate yoked learning with classical machine learning models (random forest, naive bayes and logistic regression).
    - **yoked_machine_learning_pipeline.py** contains functions for evaluating yoked learning 
    - **yoked_learning_main.py** contains the main function to run yoked learning 
    - **example boxplot/lineplot.ipynb** contains an example notebook that visualize comparisons between yoked learning, active learning, and passive learning

#### Part 2: Deep Yoked Learning
- Code and functions to evaluate yoked learning with deep learning models (MLP). 
  - Implementation methods include [MolALKit](https://github.com/RekerLab/MolALKit), default MLP parameters (ffn_num_layers: 2, ffn_hidden_size: 300, dropout: 0, batch_size:50) or optimized MLP parameters [chemprop](https://github.com/chemprop/chemprop)
  - **Single_dataset_comparison.ipynb** contains an example notebook that visualizes the output comparisons
 - Example implementation: 
```commandline
molalkit_run --data_public bace --metrics roc-auc --learning_type explorative --model_config_selector RandomForest_RDKitNorm_Config \
    --split_type scaffold_order --split_sizes 0.5 0.5 --evaluate_stride 100 --seed 0 --save_dir bace_rf_yoked_mlp --n_jobs 4 \
    --model_config_evaluators MLP_RDKitNorm_BinaryClassification_Config
```
