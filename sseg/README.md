## SHADE on Domain Generalized Semantic Segmentation

This is the implementation of SHADE on domain generalized semantic segmentation with Transformer backbone (MiT-B5). The implementation with ConvNets backbone (ResNet) is available in the [conference version](https://github.com/HeliosZhao/SHADE).

### Setup Environment
We use python 3.8.5, and pytorch 1.7.1 with cuda 11.0. 
```shell
conda create -n dgformer python=3.8
conda activate dgformer
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.7  # requires the other packages to be installed first

```
All experiments were executed on a NVIDIA RTX 3090 GPU.

### Setup Datasets

**Data Download:** We trained our model with the source domain GTAV and evaluated the model on Cityscapes, BDD-100K, and Mapillary Vistas. Please follow the [conference version](https://github.com/HeliosZhao/SHADE) to download data.


**Data Preprocessing:** Please run the following scripts to convert the label IDs to the
train IDs and to generate the class index for RCS:

```shell
python tools/convert_datasets/gta.py data/GTAV --nproc 8
python tools/convert_datasets/cityscapes.py data/CityScapes --nproc 8
python tools/convert_datasets/mapillary.py data/mapillary/validation --nproc 8
```

### Training

First, please download the MiT weights from [[Google Drive](https://drive.google.com/uc?id=1d7I50jVjtCddnhpf-lqj8-f13UyCzoW1)].

To train the source only model:

```shell
python run_experiments.py --config configs/dgformer/gta2cs_source.py
```

To train our model:

```shell
python run_experiments.py --config configs/dgformer/gta2cs_source_rsc_shade.py
```


### Testing

We use the final model to evaluate the performance on the target datasets: CityScapes, BDD100K and Mapillary.

```shell
sh scripts/test_dg.sh ${work_dir} cityscapes ## test CityScapes
sh scripts/test_dg.sh ${work_dir} bdd ## test bdd100k
sh scripts/test_dg.sh ${work_dir} mapillary ## test mapillary
```


### Framework Structure

This project is based on [mmsegmentation version 0.16.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0).
For more information about the framework structure and the config system,
please refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/index.html)
and the [mmcv documentation](https://mmcv.readthedocs.ihttps://arxiv.org/abs/2007.08702o/en/v1.3.7/index.html).


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

* [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
* [DAFormer](https://github.com/lhoyer/DAFormer)
  

