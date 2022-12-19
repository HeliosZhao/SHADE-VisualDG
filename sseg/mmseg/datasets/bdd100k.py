from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class BDD100KDataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(BDD100KDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)
