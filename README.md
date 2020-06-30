# D3Net (TNNLS2020)
Rethinking RGB-D Salient Object Detection: Models, Datasets, and Large-Scale Benchmarks, IEEE TNNLS 2020
Please refer to our website page (http://dpfan.net/d3netbenchmark/) for more details. 

## Training and Testing Sets
Our training dataset is:

https://drive.google.com/open?id=1osdm_PRnupIkM82hFbz9u0EKJC_arlQI

Our testing dataset is:

https://drive.google.com/open?id=1ABYxq0mL4lPq2F0paNJ7-5T9ST6XVHl1

## Requirement
- PyTorch>=0.4.1  
- Opencv   

## Useage
Put the three pretrained models into the created folder "eval/pretrained_model".

### Evalution:
```
python eval.py
```
## Pretrained models
-RgbdNet,RgbNet,DepthNet pretrained models can be downloaded from ( [GoogleDrive](https://drive.google.com/drive/folders/1jbZzUbgOC0XzbBEsy-Bgf3b-pvr62aWK?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1sgi0KExOv5KOfGQgXpDdqw) code: xf1h )  

## Citation
If you find this work or code is helpful in your research, please cite:
```
@article{fan2019rethinking,
  title={{Rethinking RGB-D salient object detection: Models, datasets, and large-scale benchmarks}},
  author={Fan, Deng-Ping and Lin, Zheng and Zhang, Zhao and Zhu, Menglong and Cheng, Ming-Ming},
  journal={IEEE TNNLS},
  year={2020}
}
```
