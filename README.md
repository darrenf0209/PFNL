# Supervised Causal Video Super-resolution
This repository is my final year project of Electrical and Computer Systems Engineering at Monash University. In this project, I look at an approach towards real-time video super-resolution by modifying [PFNL](http://openaccess.thecvf.com/content_ICCV_2019/html/Yi_Progressive_Fusion_Video_Super-Resolution_Network_via_Exploiting_Non-Local_Spatio-Temporal_Correlations_ICCV_2019_paper.html), a state-of-the-art network.

An [interactive dashboard](https://fyp-darrenf.herokuapp.com/) with accompanying [code](https://github.com/darrenf0209/FYP-Dash) is also presented for a partial summary of the project.

## Datasets
The datasets can be downloaded from Google Drive, [train](https://drive.google.com/open?id=1xPMYiA0JwtUe9GKiUa4m31XvDPnX7Juu) and [evaluation](https://drive.google.com/file/d/1Px0xAE2EUzXbgfDJZVR2KfG7zAk7wPZO/view?usp=sharing).

* Training: [MM522](https://github.com/psychopa4/MMCNN) 522 video sequences
* Evaluation: 20 custom sequences, mostly from documentaries, and should only be used for educational purposes.
* Testing: [Vid4](https://drive.google.com/file/d/1-Sy3t0zgbUskX1rr2Vu7oM9ssLlfIvzd/view?usp=sharing)  and [UDM10](https://drive.google.com/file/d/1IEURw2U4V9KNejw3YptPL6gWM2xLE6bq/view?usp=sharing)


## Environment
  - Python (Tested on 3.7)
  - Tensorflow (Tested on 1.12.0)
    - Tensorflow back-compatibility using tf.compat.v1 
    
## Model Summary
- Control 3,5 or 7: original PFNL network with varying input frames
- Null: previous and current frame are low resolution
- Alternative: previous frame is High resolution and current frame is low resolution

## Getting started
Unzip the training dataset to ./data/train/ and evaluation dataset to ./data/val/ for usage.

Install needed requirements

Train a model in main.py 

Test a model by calling testvideos() and appropriate calling of test function (high resolution input, low resolution input or information recycling)

## Contact
If you have questions, please open an issue here or send me an email at darrenf0209@gmail.com.

## Visual Results of 2x Super-resolution

This frame is from clap in UDM10 testing dataset. The down-sampled and blurred input to the network and corresponding network output of the Alternative model are shown.

<img src="https://github.com/darrenf0209/PFNL/blob/master/demo/clap_b4.gif" alt="Input to Network" width="500">

<img src="https://github.com/darrenf0209/PFNL/blob/master/demo/clap_alt_best.gif" alt="Output of Network" width="500">

This frame is from foliage in vid4 testing dataset. Bicubic up-sampling and the information recycling method are shown.

<img src="https://github.com/darrenf0209/PFNL/blob/master/demo/foliage_bicubic.gif" alt="Bicubic Up-sampling" width="500">

<img src="https://github.com/darrenf0209/PFNL/blob/master/demo/foliage_info_recycle.gif" alt="Information Recycling" width="500">

## Citation
If you found this project useful, please consider citing the authors of the PFNL network, which this work used as foundation.
```
@inproceedings{PFNL,
  title={Progressive Fusion Video Super-Resolution Network via Exploiting Non-Local Spatio-Temporal Correlations},
  author={Yi, Peng and Wang, Zhongyuan and Jiang, Kui and Jiang, Junjun and Ma, Jiayi},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  pages={3106-3115},
  year={2019},
}
```