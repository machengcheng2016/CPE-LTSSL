# CPE-LTSSL
Official code for "Complementary Experts for Long-tailed Semi-Supervised Learning" (AAAI 2024)

[Chengcheng Ma](https://scholar.google.com/citations?user=-Zir-A8AAAAJ&hl=en)<sup>1,2</sup>, [Ismail Elezi](https://dvl.in.tum.de/team/elezi/)<sup>3</sup>, [Jiankang Deng](https://jiankangdeng.github.io/)<sup>3</sup>, [Weiming Dong](https://scholar.google.com/citations?user=WKGx4k8AAAAJ&hl=zh-CN)<sup>1</sup>, [Changsheng Xu](https://scholar.google.com.sg/citations?user=hI9NRDkAAAAJ&hl=zh-CN)<sup>1</sup>.

<sup>1</sup> Chinese Academy of Sciences Institute of Automation (CASIA)  
<sup>2</sup> University of the Chinese Academy of Sciences (UCAS)  
<sup>3</sup> Huawei Technologies Co Ltd  

## Introduction
This repo is based on the public and widely-used codebase [USB](https://github.com/microsoft/Semi-supervised-learning).

What I've done is just adding our CPE algorithm in `semilearn/imb_algorithms/cpe`, plus some baseline algorithms which are missing in USB, such as [RDA](https://github.com/NJUyued/RDA4RobustSSL) and [ACR](https://github.com/Gank0078/ACR).

Also, I've made corresponding modifications to `semilearn/nets/` and several `__init__.py`.

## How to run
For example, on CIFAR-10-LT with $\gamma_l=\gamma_u=100$

`
CUDA_VISIBLE_DEVICES=0 python train.py --c config/classic_cv_imb/fixmatch_cpe/fixmatch_cpe_cifar10_lb1500_100_ulb3000_100_0_2_4_0.yaml
`

(Note: I know that USB supports multi-GPUs, but I still recommend you to run on single GPU, as some weird problems may occur.)

The model will be automatically evaluated every 1024 iterations during training. After training, the last two lines in `saved_models/classic_cv_imb/fixmatch_cpe_cifar10_lb1500_100_ulb3000_100_0_2_4_0/log.txt` will tell you the best accuracy. 

For example,
```
[2023-08-07 15:46:32,882 INFO] model saved: ./saved_models/classic_cv_imb/fixmatch_cpe_cifar10_lb1500_100_ulb3000_100_0_2_4_0/latest_model.pth
[2023-08-07 15:46:32,884 INFO] Model result - eval/best_acc : 0.8479
[2023-08-07 15:46:32,884 INFO] Model result - eval/best_it : 177151
```

## Results
![image](https://github.com/machengcheng2016/CPE-LTSSL/assets/22851860/efce09a9-a951-447a-87c5-75136a7a59b7)

![image](https://github.com/machengcheng2016/CPE-LTSSL/assets/22851860/225b0e9e-cbb2-45b3-a2d7-a2e23ff554ea)


I've put all the `log.txt` on the CIFAR-10-LT dataset in `saved_models`. Please take a look. I also put them on Google Drive ([link](https://drive.google.com/drive/folders/1J2npoQqSkenMUiKHtlhWKJmSOn0K0Hib?usp=sharing)).

Note that a few training process were killed halfway by me. But you can still check the `BEST_EVAL_ACC` in their `log.txt`. I promise they are real.

The reported accuracies in Table 2 and 3 in our AAAI paper are the average over three different runs (random seeds are 0/1/2). I wrote a tiny [script](https://github.com/machengcheng2016/CPE-LTSSL/blob/main/saved_models/classic_cv_imb/compute_mean_std.py) to realize quick observation and computation.


## Citation
If you find this work useful, please consider citing our paper. We provide a BibTeX entry of our paper below:
```
@misc{ma2023heads,
      title={Three Heads Are Better Than One: Complementary Experts for Long-Tailed Semi-supervised Learning}, 
      author={Chengcheng Ma and Ismail Elezi and Jiankang Deng and Weiming Dong and Changsheng Xu},
      year={2023},
      eprint={2312.15702},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Contact
Feel free to contact me via machengcheng2016@gmail.com if you have any problems about our paper or codes.
