# MuVER
This repo contains the code and pre-trained model for our EMNLP 2021 paper:       
**MuVER: Improving First-Stage Entity Retrieval with Multi-View Entity Representations**. Xinyin Ma, Yong Jiang, Nguyen Bach, Tao Wang, Zhongqiang Huang, Fei Huang, Weiming Lu

## Quick Start
### 1. Requirements
The requirements for our code are listed in requirements.txt, install the package with the following command:  
```
pip install -r requirements.txt
```
For [huggingface/transformers](https://github.com/huggingface/transformers), we tested it under version 4.1.X and 4.2.X.

### 2. Download data and model
* Data:   
We follow facebookresearch/BLINK to download and preprocess data. See [instructions](https://github.com/facebookresearch/BLINK/tree/master/examples/zeshel) about how to download and convert to BLINK format. You will get a folder with the following structure:
```
- zeshel
  | - mentions
  | - documents
  | - blink_format 
```

* Model:  
Model for zeshel can be downloaded on https://drive.google.com/file/d/1BBTue5Vmr3MteGcse-ePqplWjccqm9_A/view?usp=sharing

### 3. Use the released model to reproduce our results
* **Without View Merging**:  
```
export PYTHONPATH='.'  
CUDA_VISIBLE_DEVICES=YOUR_GPU_DEVICES python muver/multi_view/train.py 
    --pretrained_model path_to_model/bert-base 
    --dataset_path path_to_dataset/zeshel
    --bi_ckpt_path path_to_model/best_zeshel.bin 
    --max_cand_len 40 
    --max_seq_len 128
    --do_test 
    --test_mode test 
    --data_parallel 
    --eval_batch_size 16
    --accumulate_score
```


Expected Result:  

|      World       |  R@1   |  R@2   |  R@4   |  R@8   |  R@16  |  R@32  |  R@50  |  R@64  |  
|------------------|--------|--------|--------|--------|--------|--------|--------|--------|  
| forgotten_realms | 0.6208 | 0.7783 | 0.8592 | 0.8983 | 0.9342 | 0.9533 | 0.9633 | 0.9700 |  
|       lego       | 0.4904 | 0.6714 | 0.7690 | 0.8357 | 0.8791 | 0.9091 | 0.9208 | 0.9249 |  
|    star_trek     | 0.4743 | 0.6130 | 0.6967 | 0.7606 | 0.8159 | 0.8581 | 0.8805 | 0.8919 |  
|      yugioh      | 0.3432 | 0.4861 | 0.6040 | 0.7004 | 0.7596 | 0.8201 | 0.8512 | 0.8672 |  
|      total       | 0.4496 | 0.5970 | 0.6936 | 0.7658 | 0.8187 | 0.8628 | 0.8854 | 0.8969 |  

* **With View Merging**:
```
export PYTHONPATH='.'  
CUDA_VISIBLE_DEVICES=YOUR_GPU_DEVICES python muver/multi_view/train.py 
    --pretrained_model path_to_model/bert-base 
    --dataset_path path_to_dataset/zeshel
    --bi_ckpt_path path_to_model/best_zeshel.bin 
    --max_cand_len 40 
    --max_seq_len 128 
    --do_test 
    --test_mode test 
    --data_parallel 
    --eval_batch_size 16
    --accumulate_score
    --view_expansion  
    --merge_layers 4  
    --top_k 0.4
```
Expected result:   
|      World       |  R@1   |  R@2   |  R@4   |  R@8   |  R@16  |  R@32  |  R@50  |  R@64  |
|------------------|--------|--------|--------|--------|--------|--------|--------|--------|
| forgotten_realms | 0.6175 | 0.7867 | 0.8733 | 0.9150 | 0.9375 | 0.9600 | 0.9675 | 0.9708 |
|       lego       | 0.5046 | 0.6889 | 0.7882 | 0.8449 | 0.8882 | 0.9183 | 0.9324 | 0.9374 |
|    star_trek     | 0.4810 | 0.6253 | 0.7121 | 0.7783 | 0.8271 | 0.8706 | 0.8935 | 0.9030 |
|      yugioh      | 0.3444 | 0.5027 | 0.6322 | 0.7300 | 0.7902 | 0.8429 | 0.8690 | 0.8826 |
|      total       | 0.4541 | 0.6109 | 0.7136 | 0.7864 | 0.8352 | 0.8777 | 0.8988 | 0.9084 |

Optional Argument:
* --data_parallel: whether you want to use multiple gpus.
* --accumulate_score: accumulate score for each entity. Obtain a higher score but will take much time to inference.  
* --view_expansion: whether you want to merge and expand view.
* --top_k: top_k pairs are expected to merge in each layer.
* --merge_layers: the number of layers for merging.
* --test_mode: If you want to generate candidates for train/dev set, change the test_mode to train or dev, which will generate candidates outputs and save it under the directory where you save the test model.

### 4. How to train your MuVER
We provice the code to train your MuVER. Train the code with the following command:  
```
export PYTHONPATH='.'  
CUDA_VISIBLE_DEVICES=YOUR_GPU_DEVICES python muver/multi_view/train.py 
    --pretrained_model path_to_model/bert-base 
    --dataset_path path_to_dataset/zeshel
    --epoch 30 
    --train_batch_size 128 
    --learning_rate 1e-5 
    --do_train --do_eval 
    --data_parallel 
    --name distributed_multi_view
```
**Important**: Since constrastive learning relies heavily on a large batch size, as reported in our paper, we use eight v100(16g) to train our model. The hyperparameters for our best model are in `logs/zeshel_hyper_param.txt`

The code will create a directory `runtime_log` to save the log, model and the hyperparameter you used. Everytime you trained your model(with or without grid search), it will create a directory under `runtime_log/name_in_your_args/start_time`, e.g., `runtime_log/distributed_multi_view/2021-09-07-15-12-21`, to store all the checkpoints, curve for visualization and the training log.  



