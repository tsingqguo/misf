Prerequisites
	Python 3
	PyTorch >= 1.0
	NVIDIA GPU + CUDA cuDNN
	
Installation	
	Install PyTorch and dependencies from: http://pytorch.org
	
	Install python requirements: pip install -r requirements.txt

Datasets
	Download images and irregular masks
		Places2: http://places2.csail.mit.edu/
		CelebA: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
		Dunhuang: https://www.cvl.iis.u-tokyo.ac.jp/e-Heritage2019/index.php?id=challenge
		
		Irregular Masks: 
			The irregular mask dataset provided by Liu et al. can be downloaded from the their website: https://nv-adlr.github.io/publication/partialconv-inpainting
	
	Generate file list:
		cd data
		python data_list.py

Training
	Set up the checkpoints/config.yml
	python /src/train.py

Testing
	Set up the checkpoints/config.yml
	python /src/test.py