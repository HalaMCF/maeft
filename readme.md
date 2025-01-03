## File Structure

./datasets: processed and split datasets

./ensemble: major voting models to label the generated IDIs

./maeft: the environment of MAEFT

./maeft_data: scripts used to process datasets and to read data

./model: model under test

./retrain: code for retraining

./train: code for training

./utils: config of datasets

## Python Version

- python 3.8.18(preferred)

## Install Requested Python Packages

- pip install -r requirements.txt (modify torch version without GPUs)

## Testing

Set the params in the argparse:

python maeft.py (for datasets excluded Ricci and Tae)

python maeft_small.py (for Ricci and Tae)

