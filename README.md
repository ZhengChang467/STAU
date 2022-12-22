# STAU (submitted to TPAMI)

Zheng Chang,
Xinfeng Zhang,
Shanshe Wang,
Siwei Ma,
Wen Gao.

Official PyTorch Code for **"MAU: A Motion-Aware Unit for Video Prediction and
Beyond"** [[paper]](https://arxiv.org/pdf/2204.09456.pdf)

### Requirements
- PyTorch 1.7
- CUDA 11.0
- CuDNN 8.0.5
- python 3.6.7

### Installation
Create conda environment:
```bash
    $ conda create -n STAU python=3.6.7
    $ conda activate STAU
    $ pip install -r requirements.txt
    $ conda install pytorch==1.7 torchvision cudatoolkit=11.0 -c pytorch
```
Download repository:
```bash
    $ git clone git@github.com:ZhengChang467/STAU.git
```
Unzip MovingMNIST Dataset:
```bash
    $ cd data
    $ unzip mnist_dataset.zip
```
### Test
Moving MNIST
```bash
    $  python STAU_run.py --dataset mnist --is_training False
```
Bair Robot Pushing
```bash
    $ python bash_bair_test.py
```
### Train
```bash
    $ python STAU_run.py --dataset mnist --is_training True
```
Bair Robot Pushing
```bash
    $ python bash_bair_train.py
```
We plan to share the train codes for other datasets soon!
### Citation
Please cite the following paper if you feel this repository useful.
```bibtex
@article{chang2022stau,
  title={STAU: A SpatioTemporal-Aware Unit for Video Prediction and Beyond},
  author={Chang, Zheng and Zhang, Xinfeng and Wang, Shanshe and Ma, Siwei and Gao, Wen},
  journal={arXiv preprint arXiv:2204.09456},
  year={2022}
}
```
### License
See [MIT License](https://github.com/ZhengChang467/MAU/blob/master/LICENSE)

