## SHADE on Domain Generalized Object Detection

This is the implementation of SHADE on domain generalized object detection. The code is based on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0).

### Setup Environment

We use python 3.8.5, and pytorch 1.7.1 with cuda 11.0. 
```shell
conda create -n dgdet python=3.8.5
conda activate dgdet
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

Compile the cuda dependencies using following simple commands:

```shell
cd lib
python setup.py build develop
```

### Data Preparation

**Urban-scene Detection**: We use the Urban-scene Detection benchmark, including Daytime-Sunny, Night-Sunny, Dusk-Rainy, Night-Rainy, and Daytime-Foggy weather.

The dataset is available at [[Download Link](https://drive.google.com/drive/folders/1IIUnUrJrvFgPzU8D6KtV0CXa8k1eBV9B)].

File structure:
```
DGDet/
|–– daytimeclear/
|–– daytimefoggy/
|–– nightrainy/
|–– duskrainy/
|–– nightclear/
```

Download and rename the dataset, and then modify the data root in `det/lib/datasets/factory.py`.

### Pretrained Model

**NOTE**. We use Caffe pretrained ResNet101 as our backbone. You can download it from [[Dropbox](https://www.dropbox.com/s/iev3tkbz5wyyuz9/resnet101_caffe.pth?dl=0), [VT Server](https://filebox.ece.vt.edu/~jw2yang/faster-rcnn/pretrained-base-models/resnet101_caffe.pth)] and put it into `data/pretrained_model`.


### Train

```shell
python train.py --cuda --no_freeze --detect_all --color_tf --add_classifier
```


### Test

```
bash test.sh ${OUT_DIR} ${CHECK_ID}
```


### Citation

```
@inproceedings{zhao2022shade,
  title={Style-Hallucinated Dual Consistency Learning for Domain Generalized Semantic Segmentation},
  author={Zhao, Yuyang and Zhong, Zhun and Zhao, Na and Sebe, Nicu and Lee, Gim Hee},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}}

@article{zhao2022shadevdg,
  title={Style-Hallucinated Dual Consistency Learning: A Unified Framework for Visual Domain Generalization},
  author={Zhao, Yuyang and Zhong, Zhun and Zhao, Na and Sebe, Nicu and Lee, Gim Hee},
  journal={arXiv preprint arXiv:2212.09068},
  year={2022}}
```


### Acknowledgements

This project is based on the following open-source projects. We thank their
authors for making the source code publically available.

* [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0)
* [Single-DGOD](https://github.com/AmingWu/Single-DGOD)
  