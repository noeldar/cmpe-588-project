# Feature Importance-aware Attack(FIA)

This repository contains the code for the FIA paper and my modifications for SHAP values: 

**[Feature Importance-aware Transferable Adversarial Attacks](https://arxiv.org/pdf/2107.14185.pdf)  (ICCV 2021)**

## Requirements

- Python 3.6.8
- Keras 2.2.4
- Tensorflow 1.14.0
- Numpy 1.16.2
- Pillow 6.0.0
- Scipy 1.2.1

## Experiments

#### Introduction

- `vgg.py` : generating shap values for input images/masked images according to pre-trained vgg-16 model

- `attack.py` : generating adversarial images with FIA attacks for initialization of the gradients pkl files ,that are output of the vgg.py, are used

- `verify.py` : the code for evaluating generated adversarial examples on different models.

  You should download the  pretrained models from ( https://github.com/tensorflow/models/tree/master/research/slim,  https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models) before running the code. Then place these model checkpoint files in `./models_tf`.

#### Usage

##### Generate adversarial examples:

- FIA with no mask

```
python attack.py --model_name vgg_16 --attack_method FIA --layer_name vgg_16/conv2/conv2_2/Relu --ens 0 --probb 0.7 --output_dir ./adv/FIA/
```




- FIA with five masks

```
python attack.py --model_name vgg_16 --attack_method FIA --layer_name vgg_16/conv2/conv2_2/Relu --ens 5 --probb 0.7 --output_dir ./adv/FIA/
```

## Evaluate the attack success rate

```
python verify.py --ori_path ./dataset/images/ --adv_path ./adv/FIA/ --output_file ./log.csv
```

## Citing this work




```
@article{wang2021feature,
  title={Feature Importance-aware Transferable Adversarial Attacks},
  author={Wang, Zhibo and Guo, Hengchang and Zhang, Zhifei and Liu, Wenxin and Qin, Zhan and Ren, Kui},
  journal={arXiv preprint arXiv:2107.14185},
  year={2021}
}
```