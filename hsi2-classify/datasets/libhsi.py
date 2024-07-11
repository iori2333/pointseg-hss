import logging
import os
from typing import Callable, Iterable
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LIBHSIDataset(Dataset):
    CLASSES = (
        "Vegetation Plant",
        "Glass Window",
        "Brick Wall",
        "Concrete Ground",
        "Block Wall",
        "Concrete Wall",
        "Concrete Footing",
        "Concrete Beam",
        "Brick Ground",
        "Glass door",
        "Vegetation Ground",
        "Soil",
        "Metal Sheet",
        "Woodchip Ground",
        "Wood/Timber Smooth Door",
        "Tiles Ground",
        "Pebble-Concrete Beam",
        "Pebble-Concrete Ground",
        "Pebble-Concrete Wall",
        "Metal Profiled Sheet",
        "Door-plastic",
        "Wood/timber Wall",
        "Wood Frame",
        "Metal Smooth Door",
        "Concrete Window Sill",
        "Metal Profiled Door",
        "Wood/Timber Profiled Door",
    )
    PALETTE = [
        [16, 122, 0],
        [186, 174, 69],
        [78, 78, 212],
        [228, 150, 139],
        [142, 142, 144],
        [222, 222, 222],
        [138, 149, 253],
        [213, 187, 160],
        [68, 229, 12],
        [203, 185, 82],
        [3, 145, 67],
        [100, 183, 232],
        [115, 127, 195],
        [16, 108, 4],
        [180, 186, 13],
        [192, 128, 255],
        [153, 31, 34],
        [98, 184, 210],
        [32, 78, 37],
        [99, 228, 172],
        [227, 250, 98],
        [150, 169, 128],
        [111, 182, 216],
        [203, 114, 188],
        [74, 5, 187],
        [124, 134, 113],
        [150, 141, 201],
    ]

    def __init__(
        self,
        data_root: str,
        img_suffix: str = ".png",
        segmap_suffix: str = "_gray.png",
        hsi_suffix: str = ".hdr",
        pipelines: Iterable[Callable[[dict], dict]] = (),
        ignore_index: int = 255,
    ) -> None:
        self.data_root = data_root
        self.img_suffix = img_suffix
        self.segmap_suffix = segmap_suffix
        self.hsi_suffix = hsi_suffix
        self.pipelines = list(pipelines)
        self.file_infos = self.load_files()
        self.ignore_index = ignore_index
        self.class_weights = None

    def load_files(self):
        img_dir = os.path.join(self.data_root, "rgb")
        ann_dir = os.path.join(self.data_root, "labels")
        hsi_dir = os.path.join(self.data_root, "reflectance_cubes")
        superpixels_dir = os.path.join(self.data_root, "lsc")

        file_infos: list[dict[str, str]] = []
        for file in os.listdir(img_dir):
            name = file.removeprefix("rgb").removesuffix(self.img_suffix)
            if not file.endswith(self.img_suffix):
                continue
            file_info = {
                "name": name,
                "img": os.path.join(img_dir, file),
                "hsi": os.path.join(
                    hsi_dir,
                    name + self.hsi_suffix,
                ),
                "ann": os.path.join(
                    ann_dir, file.replace(self.img_suffix, self.segmap_suffix)
                ),
                "superpixels": os.path.join(
                    superpixels_dir, file.replace(self.img_suffix, ".npy")
                ),
            }
            # if not os.path.exists(file_info["ann"]):
            #     continue

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
            superpixels_path=img_info["superpixels"],
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
        # out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return out
