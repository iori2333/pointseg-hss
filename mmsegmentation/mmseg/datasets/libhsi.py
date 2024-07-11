from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class LIBHSIDataset(BaseSegDataset):
    METAINFO = dict(
        classes=("Vegetation Plant", "Glass Window", "Brick Wall",
                 "Concrete Ground", "Block Wall", "Concrete Wall",
                 "Concrete Footing", "Concrete Beam", "Brick Ground",
                 "Glass door", "Vegetation Ground", "Soil", "Metal Sheet",
                 "Woodchip Ground", "Wood/Timber Smooth Door", "Tiles Ground",
                 "Pebble-Concrete Beam", "Pebble-Concrete Ground",
                 "Pebble-Concrete Wall", "Metal Profiled Sheet",
                 "Door-plastic", "Wood/timber Wall", "Wood Frame",
                 "Metal Smooth Door", "Concrete Window Sill",
                 "Metal Profiled Door", "Wood/Timber Profiled Door"),
        palette=[[16, 122, 0], [186, 174, 69], [78, 78, 212], [228, 150, 139],
                 [142, 142, 144], [222, 222, 222], [138, 149, 253],
                 [213, 187, 160], [68, 229, 12], [203, 185, 82], [3, 145, 67],
                 [100, 183, 232], [115, 127, 195], [16, 108,
                                                    4], [180, 186, 13],
                 [192, 128, 255], [153, 31, 34], [98, 184, 210], [32, 78, 37],
                 [99, 228, 172], [227, 250, 98], [150, 169, 128],
                 [111, 182, 216], [203, 114, 188], [74, 5, 187],
                 [124, 134, 113], [150, 141, 201]])

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_gray.png',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
