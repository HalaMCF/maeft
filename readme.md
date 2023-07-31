This requirements.txt only provides the basic dependencies to run ABFT. If you want to accelerate the generation process, please intall the corresponding cuda and cudnn on your own(We use cudnn=8.2.1=cuda11.3_0 in the paper).

Getting Started as follows:

1. pip install requirements.txt
2. python abft_on.py

then you can get the details of this generation process and the whole IDIs.npy.

You can get more information of training DNNs from Neuronfair and LIMI  based on tensorflow1 or you can train your own model. We provide the datasets and models used in the paper. 