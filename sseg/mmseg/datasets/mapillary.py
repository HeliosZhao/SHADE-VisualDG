from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class MapillaryDataset(CityscapesDataset):

    def __init__(self, **kwargs):
        super(MapillaryDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='_labelTrainIds.png',
            **kwargs)

