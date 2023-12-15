The requirements.txt only provides the basic dependencies to run ABFT. To accelerate the generation process, you may install the corresponding cuda and cudnn on your own. We use cudnn=8.2.1=cuda11.3_0 in the paper.

Getting Started as follows:

1. pip install requirements.txt
2. python abft_on.py

then you can get the details of the test generation process and the whole IDIs.npy.

You can get more information about training DNNs from Neuronfair and LIMI based on tensorflow 1.0 or you can train your own model. The datasets and models used in the paper are provided in this repository. 



The results of the performance of MABFT on machine learning models is shown in experiments_ml.xlsx.