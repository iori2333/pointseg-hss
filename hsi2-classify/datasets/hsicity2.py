import logging
import os
from typing import Callable, Iterable, Sequence
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class HSICityV2Dataset(Dataset):
    CLASSES = (
        "road",
        "sidewalk",
        "building",
        "wall",
        "fence",
        "pole",
        "traffic light",
        "traffic sign",
        "vegetation",
        "terrain",
        "sky",
        "person",
        "rider",
        "car",
        "truck",
        "bus",
        "train",
        "motorcycle",
        "bicycle",
    )

    PALETTE = [
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [70, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    def __init__(
        self,
        data_root: str,
        img_suffix: str = ".jpg",
        segmap_suffix: str = "_gray.png",
        hsi_suffix: str = ".pt",
        pipelines: Iterable[Callable[[dict], dict]] = (),
        ignore_index: int = 255,
        name_list: Sequence[str] | None = None,
    ) -> None:
        self.data_root = data_root
        self.img_suffix = img_suffix
        self.segmap_suffix = segmap_suffix
        self.hsi_suffix = hsi_suffix
        self.pipelines = list(pipelines)
        self.name_list = name_list

        self.file_infos = self.load_files()
        self.ignore_index = ignore_index
        self.class_weights = torch.FloatTensor(
            [
                0.6666,
                0.9572,
                0.7315,
                0.8643,
                0.8280,
                1.9843,
                0.9995,
                0.8465,
                0.6949,
                1.0520,
                0.7580,
                0.9479,
                0.9912,
                0.6867,
                0.8538,
                0.7671,
                1.2733,
                1.9843,
                1.1129,
            ]
        )

    def load_files(self):
        img_dir = os.path.join(self.data_root, "img")
        ann_dir = os.path.join(self.data_root, "gt")
        hsi_dir = os.path.join(self.data_root, "hsd")

        file_infos: list[dict[str, str]] = []
        names = self.name_list
        if names is None:
            files = filter(lambda x: x.endswith(self.img_suffix), os.listdir(img_dir))
            names = list(
                map(
                    lambda x: x.removeprefix("rgb").removesuffix(self.img_suffix), files
                )
            )
        for name in names:
            file_info = {
                "name": name,
                "img": os.path.join(img_dir, f"rgb{name}{self.img_suffix}"),
                "hsi": os.path.join(
                    hsi_dir,
                    name + self.hsi_suffix,
                ),
                "ann": os.path.join(ann_dir, f"rgb{name}{self.segmap_suffix}"),
            }

            file_infos.append(file_info)
        logging.info(f"loaded {len(file_infos)} images")
        return file_infos

    def __len__(self):
        return len(self.file_infos)

    def prepare_img(self, idx: int):
        img_info = self.file_infos[idx]
        result = dict(
            data_root=self.data_root,
            name=img_info["name"],
            img_path=img_info["img"],
            hsi_path=img_info["hsi"],
            ann_path=img_info["ann"],
        )
        for pipeline in self.pipelines:
            result = pipeline(result)
        return result

    def __getitem__(self, index: int):
        return self.prepare_img(index)

    def put_palette(self, label: np.ndarray):
        out = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8) + 255
        for i, color in enumerate(self.PALETTE):
            out[label == i] = color
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out
