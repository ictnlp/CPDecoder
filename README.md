
## CPDecoder: Speeding Up Neural Machine Translation Decoding by Cube Pruning

Although neural machine translation has achieved promising results, it suffers from slow translation speed. The direct consequence is that a trade-off has to be made between translation quality and speed, thus its performance can not come into full play. We apply cube pruning, a popular technique to speed up dynamic programming, into neural machine translation to speed up the translation.

> Wen Zhang, Liang Huang, Yang Feng, Lei Shen and Qun Liu. Speeding Up Neural Machine Translation Decoding by Cube Pruning. In Proceedings of EMNLP, 2018. [\[paper\]](http://aclweb.org/anthology/D18-1460)[\[code\]](https://github.com/ictnlp/CPDecoder/blob/master/searchs/cp.py)

### Runtime Environment
This system has been tested in the following environment.
+ Ubuntu 16.04.1 LTS 64 bits
+ Python 3.6
+ Pytorch 1.0

### Toy Dataset
+ The training data consists of 44K sentences from the tourism and travel domain
+ Validation Set was composed of the ASR devset 1 and devset 2 from IWSLT 2005
+ Testing dataset is the IWSLT 2005 test set.

### Data Preparation
Name the file names of the datasets according to the variables in the ``wargs.py`` file  
Both sides of the training dataset and the source sides of the validation/test sets are tokenized by using the Standford tokenizer.

#### Training Dataset

+ **Source side**: ``dir_data + train_prefix + '.' + train_src_suffix``  
+ **Target side**: ``dir_data + train_prefix + '.' + train_trg_suffix``  

#### Validation Set

+ **Source side**: ``val_tst_dir + val_prefix + '.' + val_src_suffix``    
+ **Target side**:  
	+ One reference  
``val_tst_dir + val_prefix + '.' + val_ref_suffix``  
	+ multiple references  
``val_tst_dir + val_prefix + '.' + val_ref_suffix + '0'``  
``val_tst_dir + val_prefix + '.' + val_ref_suffix + '1'``  
``......``

#### Test Dataset
+ **Source side**: ``val_tst_dir + test_prefix + '.' + val_src_suffix``  
+ **Target side**:  
``for test_prefix in tests_prefix``
	+ One reference  
``val_tst_dir + test_prefix + '.' + val_ref_suffix``  
	+ multiple references  
``val_tst_dir + test_prefix + '.' + val_ref_suffix + '0'``  
``val_tst_dir + test_prefix + '.' + val_ref_suffix + '1'``  
``......``
 
### Training
Before training, parameters about training in the file ``wargs.py`` should be configured  
run ``python _main.py``

### Inference
Assume that the trained model is named ``best.model.pt``  
Before decoding, parameters about inference in the file ``wargs.py`` should be configured, we set
the option search\_mode = 2 for the cube pruning decoding
+ translate one sentence  
run ``python wtrans.py -m best.model.pt``
+ translate one file  
	+ put the test file to be translated into the path ``val_tst_dir + '/'``  
	+ run ``python wtrans.py -m best.model.pt -i test_prefix``








