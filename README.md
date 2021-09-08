# RGB-D Salient Object Detection: Models, Data Sets, and Large-Scale Benchmarks (TNNLS2021)
Rethinking RGB-D Salient Object Detection: Models, Data Sets, and Large-Scale Benchmarks, IEEE TNNLS 2021. 
Please refer to our website page (http://dpfan.net/d3netbenchmark/) for more details. 

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
## Pretrained models
-RgbdNet,RgbNet,DepthNet pretrained models can be downloaded from ( [GoogleDrive](https://drive.google.com/drive/folders/1jbZzUbgOC0XzbBEsy-Bgf3b-pvr62aWK?usp=sharing) | [BaiduYun](https://pan.baidu.com/s/1sgi0KExOv5KOfGQgXpDdqw) code: xf1h )  

## Results
<p align="center">
    <img src="D3Net-Result.jpg"/> <br/>
</p>
Results of our model on seven benchmark datasets can be found: 

Baidu Pan(https://pan.baidu.com/s/13z0ZEptUfEU6hZ6yEEISuw) 提取码: r295

Google Drive(https://drive.google.com/drive/folders/1T46FyPzi3XjsB18i3HnLEqkYQWXVbCnK?usp=sharing)

## Paper with code
https://paperswithcode.com/task/rgb-d-salient-object-detection

## RGB-D SOD Models:  <a id="RGBDmodels" class="anchor" href="#RGBDmodels" aria-hidden="true"><span class="octicon octicon-link"></span></a>  
:fire::fire::fire:Update (in 2020-08-28)
**No.** | **Year** | **Model** |**Pub.** | **Title** | **Links** 
:-: | :-: | :-: | :-  | :-  | :-: 
:fire: 97 | 2020 | D3Net  |IEEE TNNLS     | Rethinking RGB-D salient object detection: models, datasets, and large-scale benchmarks | [Paper](https://arxiv.org/pdf/1907.06781.pdf)/[Project](https://github.com/DengPingFan/D3NetBenchmark)
:fire: 96 | 2020 |JL-DCF   | arXiv (CVPR extension)   | Siamese Network for RGB-D Salient Object Detection and Beyond | [Paper](https://arxiv.org/pdf/2008.12134.pdf)/[Project](https://github.com/kerenfu/JLDCF)
:fire: 95 | 2020 |MMNet  | ACM MM  | MMNet: Multi-Stage and Multi-Scale Fusion Network for RGB-D Salient Object Detection | Paper/Project
94 | 2020 |DASNet  | ACM MM        | Is depth really necessary for salient object detection? | [Paper](https://arxiv.org/pdf/2006.00269.pdf)/[Project](http://cvteam.net/projects/2020/DASNet/)
93 | 2020 |FRDT    | ACM MM        | Feature Reintegration over Differential Treatment: A Top-down and Adaptive Fusion Network for RGB-D Salient Object Detection | Paper/[Project](https://github.com/jack-admiral/ACM-MM-FRDT)
92 | 2020 | HANet   | Appl. Sci.       | Hybrid‐Attention Network for RGB‐D Salient Object Detection | Paper/Project
91 | 2020 | DQSD   | IEEE TIP        | Depth Quality Aware Salient Object Detection | [Paper](https://arxiv.org/pdf/2008.04159.pdf)/[Project](https://github.com/qdu1995/DQSD)
90 | 2020 | DQAM   | arXiv         | Knowing Depth Quality In Advance: A Depth Quality Assessment Method For RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2008.04157.pdf)/Project
89 | 2020 | SDSFNet| IEEE TIP      | Improved Saliency Detection in RGB-D Images Using Two-phase Depth Estimation and Selective Deep Fusion | [Paper](http://probb268dca.pic5.ysjianzhan.cn/upload/TIP20_FHR.pdf)/Project
88 | 2020 | ERLF   | IEEE TIP      | Data-Level Recombination and Lightweight Fusion Scheme for RGB-D Salient Object Detection | [Paper](http://probb268dca.pic5.ysjianzhan.cn/upload/TIP20_WXH_q02i.pdf)/[Project](https://github.com/XueHaoWang-Beijing/DRLF)
87 | 2020 | MCINet | arXiv         | MCINet: Multi-level Cross-modal Interaction Network for RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2007.14352.pdf)/Project
86 | 2020 | PGAR   | ECCV          | Progressively Guided Alternate Refinement Network for RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2008.07064.pdf)/[Project](https://github.com/ShuhanChen/PGAR_ECCV20)
85 | 2020 | ATSA   | ECCV          | Asymmetric Two-Stream Architecture for Accurate RGB-D Saliency Detection	| [Paper](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730375.pdf)/[Project](https://github.com/sxfduter/ATSA)
:fire: 84 | 2020 | BBS-Net| ECCV          | BBS-Net: RGB-D Salient Object Detection with a Bifurcated Backbone Strategy Network | [Paper](https://arxiv.org/pdf/2007.02713.pdf)/[Project](https://github.com/zyjwuyan/BBS-Net)
83 | 2020 | CoNet  | ECCV          | Accurate RGB-D Salient Object Detection via Collaborative Learning | [Paper](https://arxiv.org/pdf/2007.11782.pdf)/[Project](https://github.com/jiwei0921/CoNet)
82 | 2020 | DANet  | ECCV          | A Single Stream Network for Robust and Real-time RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2007.06811.pdf)/[Project](https://github.com/Xiaoqi-Zhao-DLUT/DANet-RGBD-Saliency)
81 | 2020 | CMMS   | ECCV          | RGB-D salient object detection with cross-modality modulation and selection | [Paper](https://arxiv.org/pdf/2007.07051.pdf)/[Project](https://github.com/Li-Chongyi/cmMS-ECCV20)
80 | 2020 | CAS-GNN| ECCV          | Cascade graph neural networks for RGB-D salient object detection | [Paper](https://arxiv.org/pdf/2008.03087.pdf)/[Project](https://github.com/LA30/Cas-Gnn)
79 | 2020 | HDFNet | ECCV          | Hierarchical Dynamic Filtering Network for RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2007.06227.pdf)/[Project](https://github.com/lartpang/HDFNet)
78 | 2020 | CMWNet | ECCV          | Cross-modal weighting network for RGB-D salient object detection | [Paper](https://arxiv.org/pdf/2007.04901.pdf)/[Project](https://github.com/MathLee/CMWNet)
:fire: 77 | 2020 | UC-Net | CVPR (Best Paper Nomination)          | UC-Net: Uncertainty Inspired RGB-D Saliency Detection via Conditional Variational Autoencoders | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_UC-Net_Uncertainty_Inspired_RGB-D_Saliency_Detection_via_Conditional_Variational_Autoencoders_CVPR_2020_paper.pdf)/[Project](https://github.com/JingZhang617/UCNet)
76 | 2020 | S2MA   | CVPR          | Learning selective self-mutual attention for RGB-D saliency detection | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Learning_Selective_Self-Mutual_Attention_for_RGB-D_Saliency_Detection_CVPR_2020_paper.pdf)/[Project](https://github.com/nnizhang/S2MA)
75 | 2020 | SSF    | CVPR          | Select, supplement and focus for RGB-D saliency detection | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Select_Supplement_and_Focus_for_RGB-D_Saliency_Detection_CVPR_2020_paper.pdf)/[Project](https://github.com/OIPLab-DUT/CVPR_SSF-RGBD)
74 | 2020 | A2dele | CVPR          | A2dele: Adaptive and Attentive Depth Distiller for Efficient RGB-D Salient Object Detection | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Piao_A2dele_Adaptive_and_Attentive_Depth_Distiller_for_Efficient_RGB-D_Salient_CVPR_2020_paper.pdf)/[Project](https://github.com/OIPLab-DUT/CVPR2020-A2dele)
73 | 2020 | JL-DCF | CVPR          | JL-DCF: Joint learning and densely-cooperative fusion framework for RGB-D salient object detection | [Paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Fu_JL-DCF_Joint_Learning_and_Densely-Cooperative_Fusion_Framework_for_RGB-D_Salient_CVPR_2020_paper.pdf)/[Project](https://github.com/kerenfu/JLDCF)
72 | 2020 | RGBS   |MTAP           | Salient object detection for RGB-D images by generative adversarial network | [Paper](https://link.springer.com/article/10.1007/s11042-020-09188-8)/Project
71 | 2020 | GFNe   |IEEE SPL       | GFNet: Gate fusion network with res2net for detecting salient objects in RGB-D images | [Paper](https://ieeexplore.ieee.org/document/9090350)/Project
70 | 2020 | SDF   | IEEE TIP       | Improved saliency detection in RGB-D images using two-phase depth estimation and selective deep fusion | [Paper](https://ieeexplore.ieee.org/document/8976428)/Project
69 | 2020 | ICNet | IEEE TIP       | ICNet: Information Conversion Network for RGB-D Based Salient Object Detection| [Paper](https://ieeexplore.ieee.org/document/9024241)/[Project](https://github.com/MathLee/ICNet-for-RGBD-SOD)
68 | 2020 |Triple-Net | IEEE SPL   | Triple-complementary network for RGB-D salient object detection| [Paper](https://ieeexplore.ieee.org/document/9076277)/Project
67 | 2020 |ASIF-Net | IEEE TCYB    | ASIF-Net: Attention steered interweave fusion network for RGB-D salient object detection| [Paper](https://ieeexplore.ieee.org/document/8998588)/[Project](https://github.com/Li-Chongyi/ASIF-Net)
66 | 2020 |BiANet | IEEE TIP       | Bilateral Attention Network for RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2004.14582.pdf)/Project
65 | 2020 |PGHF   | IEEE Access    | Multi-modal weights sharing and hierarchical feature fusion for rgbd salient object detection | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8981965)/Project
64 | 2020 |cmSalGAN | IEEE TMM     | cmSalGAN: RGB-D Salient Object Detection with Cross-View Generative Adversarial Networks | [Paper](https://arxiv.org/pdf/1912.10280.pdf)/Project
63 | 2020 | CoCNN | PR             | CoCNN: RGB-D deep fusion for stereoscopic salient object detection | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320320301321)/Project
62 | 2020 | GFNet | Neurocomputing | A cross-modal adaptive gated fusion generative adversarial network for RGB-D salient object detection | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231220300904)/Project
61 | 2020 | AttNet| IVC            | Attention-guided RGBD saliency detection using appearance information | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0262885620300202)/Project
60 | 2020 | SSDF   |arXiv          | Synergistic saliency and depth prediction for RGB-D saliency detection | [Paper](https://arxiv.org/pdf/2007.01711.pdf)/Project
59 | 2020 |DPANet | arXiv          | DPANet: Depth Potentiality-Aware Gated Attention Network for RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/2003.08608.pdf)/Project
58 | 2019 | DSD   | JVCIR          | Depth-aware saliency detection using convolutional neural networks | [Paper](https://www.sciencedirect.com/science/article/pii/S104732031930118X)/Project
57 | 2019 | DMRA  | ICCV           | Depth-induced Multi-scale Recurrent Attention Network for Saliency Detection | [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Piao_Depth-Induced_Multi-Scale_Recurrent_Attention_Network_for_Saliency_Detection_ICCV_2019_paper.pdf)/[Project](https://github.com/jiwei0921/DMRA)
56 | 2019 | CPFP  | CVPR           | Contrast Prior and Fluid Pyramid Integration for RGBD Salient Object Detection | [Paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Contrast_Prior_and_Fluid_Pyramid_Integration_for_RGBD_Salient_Object_CVPR_2019_paper.pdf)/[Project](https://github.com/JXingZhao/ContrastPrior)
55 | 2019 | EPM   | IEEE Access    | Co-saliency detection for rgbd images based on effective propagation mechanism | [Paper](https://ieeexplore.ieee.org/document/8849990)/Project
54 | 2019 | AFNet | IEEE Access    | Adaptive Fusion for RGB-D Salient Object Detection | [Paper](https://arxiv.org/pdf/1901.01369.pdf)/[Project](https://github.com/Lucia-Ningning/Adaptive_Fusion_RGBD_Saliency_Detection)
53 | 2019 | LSF   | arXiv          | CNN-based RGB-D Salient Object Detection: Learn, Select and Fuse | [Paper](https://arxiv.org/pdf/1909.09309.pdf)/Project
52 | 2019 | DGT   | IEEE TCYB      | Going from RGB to RGBD saliency: A depth-guided transformation model | [Paper](https://www.researchgate.net/publication/335360400_Going_From_RGB_to_RGBD_Saliency_A_Depth-Guided_Transformation_Model)/[Project](https://rmcong.github.io/proj_RGBD_sal_DTM_tcyb.html)
51 | 2019 | DCMF  | IEEE TCYB      | Discriminative cross-modal transfer learning and densely cross-level feedback fusion for RGB-D salient object detection | [Paper](https://ieeexplore.ieee.org/document/8820129)/Project
50 | 2019 | TANet | IEEE TIP       | Three-stream attention-aware network for RGB-D salient object detection | [Paper](https://ieeexplore.ieee.org/document/8603756)/Project
49 | 2019 | DCA   | IEEE TIP       | Saliency detection via depth-induced cellular automata on light field| [Paper](https://ieeexplore.ieee.org/document/8866752)/Project
48 | 2019 | MMCI  | PR             | Multi-modal fusion network with multi-scale multi-path and cross-modal interactions| [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0031320318303054)/Project
47 | 2019 | PDNet | ICME           | Prior-model guided depth-enhanced network for salient object detection| [Paper](https://arxiv.org/pdf/1803.08636.pdf)/[Project](https://github.com/cai199626/PDNet)
46 | 2019 | CAFM  | IEEE TSMC      | Global and Local-Contrast Guides Content-Aware Fusion for RGB-D Saliency Prediction | [Paper](https://ieeexplore.ieee.org/document/8941002)/Project
45 | 2019 | DIL   | MTAP           | Salient object segmentation based on depth-aware image layering | [Paper](https://link.springer.com/article/10.1007/s11042-018-6736-4)/Project
44 | 2019 | TSRN  | ICIP           | Two-stream refinement network for RGB-D saliency detection | [Paper](https://ieeexplore.ieee.org/document/8803653)/Project
43 | 2019 | MLF   | SPL            | RGB-D salient object detection by a CNN with multiple layers fusion | [Paper](https://ieeexplore.ieee.org/document/8638984)/Project
42 | 2019 | SSRC  | Neurocomputing | Salient object detection for RGB-D image by single stream recurrent convolution neural network | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231219309403)/Project
41 | 2018 | CDB   | Neurocomputing | Stereoscopic saliency model using contrast and depth-guided-background prior | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231217317034)/Project
40 | 2018 | ACCF  | IROS           | Attention-Aware Cross-Modal Cross-Level Fusion Network for RGB-D Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/8594373)/Project
39 | 2018 | SCDL  | ICDSP          | Rgbd salient object detection using spatially coherent deep learning framework | [Paper](https://ieeexplore.ieee.org/document/8631584)/Project
38 | 2018 | PCF   | CVPR           | Progressively complementarityaware fusion network for RGB-D salient object detection | [Paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Chen_Progressively_Complementarity-Aware_Fusion_CVPR_2018_paper.pdf)/[Project](https://github.com/haochen593/PCA-Fuse_RGBD_CVPR18)
37 | 2018 | CTMF  | IEEE TCYB      | CNNs-based RGB-D saliency detection via cross-view transfer and multiview fusion | [Paper](https://ieeexplore.ieee.org/document/8091125)/[Project](https://github.com/haochen593/CTMF)
36 | 2018 | ICS   | IEEE TIP       | Co-saliency detection for RGBD images based on multi-constraint feature matching and cross label propagation | [Paper](https://arxiv.org/pdf/1710.05172.pdf)/Project
35 | 2018 | HSCS  | IEEE TMM       | HSCS: Hierarchical sparsity based co-saliencydetection for RGBD images | [Paper](https://arxiv.org/pdf/1811.06679.pdf)/[Project](https://github.com/rmcong/Results-for-2018TMM-HSCS)
34 | 2017 | ISC   | SIVP           | An integration of bottom-up and top-down salient cueson rgb-d data: saliency from objectness versus non-objectness | [Paper](https://arxiv.org/pdf/1807.01532.pdf)/Project
33 | 2017 | MCLP  | IEEE TCYB      | An iterative co-saliency framework for RGBD images | [Paper](https://arxiv.org/pdf/1711.01371.pdf)/Project
32 | 2017 | DF    | IEEE TIP       | RGBD Salient Object Detection via Deep Fusion | [Paper](https://arxiv.org/pdf/1607.03333.pdf)/[Project](https://pan.baidu.com/s/1Y-PqAjuH9xREBjfl7H45HA)
31 | 2017 | MDSF  | IEEE TIP       | Depth-Aware Salient Object Detection and Segmentation via Multiscale Discriminative Saliency Fusion and Bootstrap Learning | [Paper](https://ieeexplore.ieee.org/document/7938352)/[Project](https://github.com/ivpshu/Depth-aware-salient-object-detection-and-segmentation-via-multiscale-discriminative-saliency-fusion-)
30 | 2017 | MFF   | IEEE SPL       | RGB-D saliency object detection via minimum barrier distance transformand saliency fusion | [Paper](https://wanganzhi.github.io/papers/SPL17.pdf)/Project
29 | 2017 | TPF   | ICCVW          | A Three-Pathway Psychobiological Framework of Salient Object Detection Using Stereoscopic Technology | [Paper](https://ieeexplore.ieee.org/document/8265566)/Project
28 | 2017 | CDCP  | ICCVW          | An innovative salient object detection using center-dark channel prior | [Paper](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w22/Zhu_An_Innovative_Salient_ICCV_2017_paper.pdf)/[Project](https://github.com/ChunbiaoZhu/ACVR2017)
27 | 2017 | BED   | ICCVW          | Learning RGB-D Salient Object Detection using background enclosure, depth contrast, and top-down features | [Paper](https://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w40/Shigematsu_Learning_RGB-D_Salient_ICCV_2017_paper.pdf)/[Project](https://github.com/sshige/rgbd-saliency)
26 | 2017 | MFLN  | ICCVS          | RGB-D Saliency Detection by Multi-stream Late Fusion Network | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-68345-4_41)/Project
25 | 2017 | M3Net | IROS           | M3Net: Multi-scale multi-path multi-modal fusion network and example application to RGB-D salient object detection | [Paper](https://ieeexplore.ieee.org/abstract/document/8206370)/Project
24 | 2017 | HOSO  | DICTA          | HOSO: Histogram of Surface Orientation for RGB-D Salient Object Detection | [Paper](https://ieeexplore.ieee.org/document/8227440)/Project
23 | 2016 | GM    | ACCV           | Visual Saliency detection for RGB-D images with generative mode | [Paper](https://link.springer.com/chapter/10.1007/978-3-319-54193-8_2)/Project
22 | 2016 | DSF   | ICASSP         | Depth-aware saliency detection using discriminative saliency fusion | [Paper](https://ieeexplore.ieee.org/document/7471952)/Project
21 | 2016 | DCI   | ICASSP         | Saliency analysis based on depth contrast increased | [Paper](http://sites.nlsde.buaa.edu.cn/~shenghao/Download/publications/2016/9.Saliency%20analysis%20based%20on%20depth%20contrast%20increased.pdf)/Project
20 | 2016 | BF    | ICPR           | RGB-D saliency detection under Bayesian framework | [Paper](https://ieeexplore.ieee.org/document/7899911)/Project
19 | 2016 | DCMC  | IEEE SPL       | Saliency detection for stereoscopic images based on depth confidence analysis and multiple cues fusion  | [Paper](https://ieeexplore.ieee.org/document/7457641)/[Project](https://github.com/rmcong/Code-for-DCMC-method)
18 | 2016 | SE    | ICME           | Salient object detection for RGB-D image via saliency evolution | [Paper](https://ieeexplore.ieee.org/document/7552907)/Project
17 | 2016 | LBE   | CVPR           | Local Background Enclosure for RGB-D Salient Object Detection| [Paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/app/S10-09.pdf)/[Project](http://users.cecs.anu.edu.au/~u4673113/lbe.html)
16 | 2016 | PRC   | IEEE Access    | Improving RGBD Saliency Detection Using Progressive Region Classification and Saliency Fusion| [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7762806)/Project
15 | 2015 | SF    | CAC            | Selective features for RGB-D saliency | [Paper](https://ieeexplore.ieee.org/document/7382554)/Project
14 | 2015 | MGMR  | ICIP           | RGB-D saliency detection via mutual guided manifold ranking | [Paper](https://ieeexplore.ieee.org/document/7350882)/Project
13 | 2015 | SRD   | ICRA           | Salient Regions Detection for Indoor Robots using RGB-D Data | [Paper](http://www.cogsys.cs.uni-tuebingen.de/publikationen/2015/Jiang_ICRA15.pdf)/Project
12 | 2015 | DIC   | TVC            | Depth incorporating with color improves salient object detection | [Paper](https://link.springer.com/article/10.1007/s00371-014-1059-6)/Project
11 | 2015 | SFP   | ICIMCS         | Salient object detection in RGB-D image based on saliency fusion and propagation | [Paper](https://dl.acm.org/doi/10.1145/2808492.2808551)/Project
10 | 2015 | GP    | CVPRW          | Exploiting global priors for RGB-D saliency detection | [Paper](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W14/papers/Ren_Exploiting_Global_Priors_2015_CVPR_paper.pdf)/[Project](https://github.com/JianqiangRen/Global_Priors_RGBD_Saliency_Detection)
09 | 2014 | ACSD  | ICIP           | Depth saliency based on anisotropic center-surround difference | [Paper](https://projet.liris.cnrs.fr/imagine/pub/proceedings/ICIP-2014/Papers/1569913831.pdf)/[Project](https://github.com/HzFu/DES_code)
08 | 2014 | DESM  | ICIMCS         | Depth Enhanced Saliency Detection Method | [Paper](http://dpfan.net/wp-content/uploads/DES_dataset_ICIMCS14.pdf)/Project
07 | 2014 | LHM   | ECCV           | RGBD Salient Object Detection: A Benchmark and Algorithms | [Paper](http://dpfan.net/wp-content/uploads/NLPR_dataset_ECCV14.pdf)/[Project](https://sites.google.com/site/rgbdsaliency/code)
06 | 2014 | SRDS  | ICDSP          | Salient region detection for stereoscopic images | [Paper](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6900706)/Project 
05 | 2013 | SOS   | Neurocomputing | Depth really Matters: Improving Visual Salient Region Detection with Depth | [Paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231213002981)/Project 
04 | 2013 | RC    | BMVC           | Depth really Matters: Improving Visual Salient Region Detection with Depth | [Paper](http://cdn.iiit.ac.in/cdn/cvit.iiit.ac.in/images/ConferencePapers/2013/cv_deepth-really.pdf)/Project 
03 | 2013 | LS    | BMVC           | An In Depth View of Saliency                                     | [Paper](http://www.cs.utah.edu/~thermans/papers/ciptadi-bmvc2013.pdf)/Project 
02 | 2012 | RCM   | ICCSE          | Depth combined saliency detection based on region contrast model | [Paper](https://ieeexplore.ieee.org/document/6295184)/Project 
01 | 2012 | DM    | ECCV           | Depth matters: Influence of depth cues on visual saliency        | [Paper](https://link.springer.com/content/pdf/10.1007/978-3-642-33709-3_8.pdf)/Project 

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
  year={2020}
}
@article{zhou2020rgbd,
  title={RGB-D Salient Object Detection: A Survey},
  author={Zhou, Tao and Fan, Deng-Ping and Cheng, Ming-Ming and Shen, Jianbing and Shao, Ling},
  journal={arXiv preprint arXiv:2008.00230},
  year={2020}
}
```
