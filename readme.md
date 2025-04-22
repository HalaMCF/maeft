## File Structure

datasets includes the processed and split datasets

ensemble includes the major voting models (extract ensemble.zip) to label the generated IDIs, and the code for training these models

maeft includes the environment of MAEFT. The maximum number of tests (denoted as to_divide) is hard coded in the ./maeft/env/environment.py line 58-61.

maeft_data includes scripts used to process datasets and to read data

model includes models under test used in the experiment.

retrain includes codes for retraining. 

train includes codes for training a MLP/CatBoost/FT-Transformer model.

utils includes configs of datasets and scripts used to determine whether the generated sample is discriminate or not. 

## Python Version

- python 3.8.18 (preferred)

## Install Requested Python Packages

- pip install -r requirements.txt (modify torch version without GPUs)

## Testing

Here are the sensitive indices for 10 datasets we used to evaluate our tool:

Census(Adult): 0 for age, 7 for race, 8 for sex
Credit: 8 for sex, 12 for age
Bank: 0 for age
Meps: 1 for age, 2 for sex, 3 for race
Ricci: 3 for race
Tae: 0 for whether_is_native_or_not
Math & Por: 1 for sex, 2 for age
COMPAS: 3 for race
Oulad: 2 for sex



Set the params in the argparse (RQ1):

`python maeft.py --dataset oulad --model_struct ft --task_type multiclass --d_out 3 --sensitive 2 (for datasets excluded Ricci and Tae)`

`python maeft.py --dataset math --model_struct ft --task_type regression --d_out 1 --sensitive 2 (for regression tasks)`

`python maeft_small.py --dataset tae --model_struct ft --task_type binclass --d_out 1 --sensitive 0 (for Ricci and Tae)`

Note: d_out is a parameter of FT-Transformer(ft), it is only related to the output dimension of ft.



Set the params in the argparse (RQ2):

Note: Need to fill in the folder where the file is located by yourself.

line 40 for retrain_cat_rg.py

line 56 for retrain_cat.py

line 41 for retrain_ft_rg.py

line 58 for retrain_ft.py

line 47 for retrain_mlp_rg.py

line 63 for retrain_mlp.py

`python retrain_mlp.py --dataset oulad --method maeft --task_type multiclass --sensitive 2 (for MLP models and classification tasks)`

`python retrain_mlp_rg.py --dataset math --method maeft --sensitive 2 (for MLP models and regression tasks)`



Set the params in the argparse (RQ5)

`python maeft.py --dataset census --model_struct ft --task_type binclass --d_out 1 --sensitive 0,7,8 `

`python maeft.py --dataset math --model_struct ft --task_type regression --d_out 1 --sensitive 1,2 `

## Evaluation

To calculate CDE, PCC and Pα (RQ3), please refer to https://github.com/amazon-science/tabsyn

To calculate IFs (RQ4), `python retrain_mlp.py` (retrain_mlp is in ./retrain folder)