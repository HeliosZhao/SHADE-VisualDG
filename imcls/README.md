## SHADE on Domain Generalized Image Classification

This is the implementation of SHADE on domain generalized image classification. 

### Setup Environment

We use python 3.8.5, and pytorch 1.7.1 with cuda 11.0. 
```shell
conda create -n dgcls python=3.8.5
conda activate dgcls
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```


### Data Preparation

**PACS**: We use the PACS benchmark.

The dataset is available at [[Download Link](https://drive.google.com/file/d/1m4X4fROCCXMO0lRLrr6Zz9Vb3974NWhE/view)].

File structure:
```
pacs/
|–– images/
|–– splits/
```


### Run

ERM+SHADE:
```shell
bash scripts/train_erm.sh
```

RSC+SHADE:
```shell
bash scripts/train_rsc.sh
```

L2D+SHADE:
```shell
bash scripts/train_l2d.sh
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

* [RSC](https://github.com/DeLightCMU/RSC)
* [Learning_to_diversify](https://github.com/BUserName/Learning_to_diversify)
  