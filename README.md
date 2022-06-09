# RGB-D Salient Object Detection: Models, Data Sets, and Large-Scale Benchmarks (TNNLS2021)
Rethinking RGB-D Salient Object Detection: Models, Data Sets, and Large-Scale Benchmarks, IEEE TNNLS 2021. 

### 0.1. :fire: NEWS :fire:
- [2020/08/02] :boom: Release the training code.

<p align="center">
    <img src="D3Net-TNNLS20.png"/> <br/>
    <em> 
Figure 1: Illustration of the proposed D3Net. In the training stage (Left), the input RGB and depth images are processed with three parallel sub-networks, e.g., RgbNet, RgbdNet, and DepthNet. The three sub-networks are based on a same modified structure of Feature Pyramid Networks (FPN) (see § IV-A for details). We introduced these sub-networks to obtain three saliency maps (i.e., Srgb, Srgbd, and Sdepth) which considered both coarse and fine details of the input. In the test phase (Right), a novel depth depurator unit (DDU) (§ IV-B) is utilized for the first time in this work to explicitly discard (i.e., Srgbd) or keep (i.e., Srgbd) the saliency map introduced by the depth map. In the training/test phase, these components form a nested structure and are elaborately designed (e.g., gate connection in DDU) to automatically learn the salient object from the RGB image and Depth image jointly.
    </em>
</p>

## Training and Testing Sets
Our training dataset is:

https://drive.google.com/open?id=1osdm_PRnupIkM82hFbz9u0EKJC_arlQI

Our testing dataset is:

https://drive.google.com/open?id=1ABYxq0mL4lPq2F0paNJ7-5T9ST6XVHl1

## Requirement
- PyTorch>=0.4.1  
- Opencv   


## Train:
Put the three datasets 'NJU2K_TRAIN', 'NLPR_TRAIN','NJU2K_TEST' into the created folder "dataset".

Put the vgg-pretrained model 'vgg16_feat.pth' ( [GoogleDrive](https://drive.google.com/file/d/1SXOV-DKnnqFD_b9yxJCIzdSkU7qiHh1X/view?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/17qaLM3nbgR_eGehSK-SOrA) code: zsxh )  into the created folder "model".
```
python train.py --net RgbNet
python train.py --net RgbdNet
python train.py --net DepthNet
```

## Evalution:
Put the three pretrained models into the created folder "eval/pretrained_model".
```
python eval.py
```

Toolbox (updated in 2022/06/09): [Baidu: i09j] (https://pan.baidu.com/s/1ArnPZ4OwP67NR71OWYjitg) | [Google] (https://drive.google.com/file/d/1I4Z7rA3wefN7KeEQvkGA92u99uXS_aI_/view?usp=sharing)
## Pretrained models
-RgbdNet,RgbNet,DepthNet pretrained models can be downloaded from ( [GoogleDrive](https://drive.google.com/drive/folders/1jbZzUbgOC0XzbBEsy-Bgf3b-pvr62aWK?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1sgi0KExOv5KOfGQgXpDdqw) code: xf1h )  

## Results
<p align="center">
    <img src="D3Net-Result.jpg"/> <br/>
</p>
Results of our model on seven benchmark datasets can be found: 

Baidu Pan(https://pan.baidu.com/s/13z0ZEptUfEU6hZ6yEEISuw) 提取码: r295

Google Drive(https://drive.google.com/drive/folders/1T46FyPzi3XjsB18i3HnLEqkYQWXVbCnK?usp=sharing)

## Paper list
https://github.com/taozh2017/RGBD-SODsurvey

## Paper with code
https://paperswithcode.com/task/rgb-d-salient-object-detection


## RGB-D SOD Datasets:  <a id="datasets" class="anchor" href="#datasets" aria-hidden="true"><span class="octicon octicon-link"></span></a>  


**No.** |**Dataset** | **Year** | **Pub.** |**Size** | **#Obj.** | **Types** | **Resolution** | **Download**
:-: | :-: | :-: | :-  | :-  | :-:| :-: | :-: | :-:
1   | [**STERE**](http://dpfan.net/wp-content/uploads/STERE_dataset_CVPR12.pdf)   |2012 |CVPR   | 1000 | ~One       |Internet             | [251-1200] * [222-900] | [link](http://dpfan.net/d3netbenchmark/)
2   | [**GIT**](http://www.bmva.org/bmvc/2013/Papers/paper0112/abstract0112.pdf)  |2013 |BMVC   | 80   | Multiple  |Home environment     | 640 * 480 | [link](http://dpfan.net/d3netbenchmark/)
3   | [**DES**](http://dpfan.net/wp-content/uploads/DES_dataset_ICIMCS14.pdf)     |2014 |ICIMCS | 135  | One       |Indoor               | 640 * 480 | [link](http://dpfan.net/d3netbenchmark/)
4   | [**NLPR**](http://dpfan.net/wp-content/uploads/NLPR_dataset_ECCV14.pdf)     |2014 |ECCV   | 1000 | Multiple  |Indoor/outdoor       | 640 * 480, 480 * 640 | [link](http://dpfan.net/d3netbenchmark/)
5   | [**LFSD**](http://dpfan.net/wp-content/uploads/LFSD_dataset_CVPR14.pdf)     |2014 |CVPR   | 100  | One       |Indoor/outdoor       | 360 * 360 | [link](http://dpfan.net/d3netbenchmark/)
6   | [**NJUD**](http://dpfan.net/wp-content/uploads/NJU2K_dataset_ICIP14.pdf)     |2014 |ICIP   | 1985 | ~One       |Moive/internet/photo | [231-1213] * [274-828] | [link](http://dpfan.net/d3netbenchmark/)
7   | [**SSD**](http://dpfan.net/wp-content/uploads/SSD_dataset_ICCVW17.pdf)      |2017 |ICCVW  | 80   | Multiple  |Movies               | 960 *1080  | [link](http://dpfan.net/d3netbenchmark/)
8   | [**DUT-RGBD**](https://openaccess.thecvf.com/content_ICCV_2019/papers/Piao_Depth-Induced_Multi-Scale_Recurrent_Attention_Network_for_Saliency_Detection_ICCV_2019_paper.pdf) |2019 |ICCV   | 1200 | Multiple  |Indoor/outdoor       | 400 * 600 | [link](http://dpfan.net/d3netbenchmark/)
9   | [**SIP**](http://dpfan.net/wp-content/uploads/SIP_dataset_TNNLS20.pdf)     |2020 |TNNLS  | 929  | Multiple  |Person in wild       | 992 * 774 | [link](http://dpfan.net/d3netbenchmark/)

## Citation
If you find this work or code is helpful in your research, please cite:
```
@article{fan2019rethinking,
  title={{Rethinking RGB-D salient object detection: Models, datasets, and large-scale benchmarks}},
  author={Fan, Deng-Ping and Lin, Zheng and Zhang, Zhao and Zhu, Menglong and Cheng, Ming-Ming},
  journal={IEEE TNNLS},
  year={2021}
}
@article{zhou2021rgbd,
  title={RGB-D Salient Object Detection: A Survey},
  author={Zhou, Tao and Fan, Deng-Ping and Cheng, Ming-Ming and Shen, Jianbing and Shao, Ling},
  journal={CVMJ},
  year={2021}
}
```
