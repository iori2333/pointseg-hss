from typing import Sequence
from typing_extensions import override
import numpy as np
import torch
import torch.utils.data
import cv2
import spectral


class LoadHSIFromFile:
    @staticmethod
    def load_hsi(file_path: str):
        data_dict = torch.load(file_path)
        height = data_dict["height"]
        width = data_dict["width"]
        SR = data_dict["SR"]
        average = data_dict["average"]
        coeff = data_dict["coeff"]
        scoredata = data_dict["scoredata"]

        temp = torch.mm(scoredata, coeff)
        data = (temp + average).reshape((height, width, SR))

        return data.numpy()

    def __call__(self, results):
        hsi_path = results["hsi_path"]
        data = self.load_hsi(hsi_path)
        results["hsi"] = data
        results["hsi_shape"] = data.shape
        return results


class LoadENVIFromFile(LoadHSIFromFile):
    @staticmethod
    @override
    def load_hsi(file_path: str):
        data = spectral.open_image(file_path)
        image_cube = data.asarray()
        image_cube = np.rot90(image_cube, 3)
        return image_cube


class LoadRGBFromFile:
    def __call__(self, results):
        img_path = results["img_path"]
        ann_path = results["ann_path"]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        ann = cv2.imread(ann_path, cv2.IMREAD_GRAYSCALE)
        # img = img.transpose(2, 0, 1)
        results["img"] = (img / 255).astype(np.float32)
        results["ann"] = ann
        results["img_shape"] = img.shape
        results["img_resolution"] = ann.shape

        return results


class GenerateHSICube:
    def __init__(self, samples: int, width: int, data_key: str = "hsi") -> None:
        assert width % 2 == 1, "cube width must be odd"
        self.samples = samples
        self.width = width
        self.data_key = data_key

    def generate_cubes(self, data: np.ndarray, label: np.ndarray):
        slide_data = np.lib.stride_tricks.sliding_window_view(
            data, (self.width, self.width), axis=(0, 1)  # type: ignore
        )
        slide_label = np.lib.stride_tricks.sliding_window_view(
            label, (self.width, self.width), axis=(0, 1)  # type: ignore
        )
        h, w, *_ = slide_data.shape
        half_width = (self.width - 1) // 2
        mask = slide_label[:, :, half_width, half_width] != 255
        actual_mask = np.zeros_like(mask)
        # choose self.samples samples from mask
        count = 0
        max_count = mask.sum()
        while count < self.samples:
            x = np.random.randint(0, h)
            y = np.random.randint(0, w)
            if mask[x, y] and not actual_mask[x, y]:
                actual_mask[x, y] = True
                count += 1
        out = slide_data[actual_mask]
        out_l = slide_label[actual_mask, half_width, half_width]
        return out, out_l

    def __call__(self, results):
        data = results[self.data_key]
        ann = results["ann"]
        cubes, labels = self.generate_cubes(data, ann)
        # loader = torch.utils.data.DataLoader(
        #     torch.utils.data.TensorDataset(
        #         torch.from_numpy(cubes).float(), torch.from_numpy(labels).long()
        #     ),
        #     batch_size=self.loader_bs,
        #     shuffle=True,
        # )
        results["cubes"] = cubes
        results["labels"] = labels
        return results


class BalancedHSICubes(GenerateHSICube):
    def __init__(
        self, num_classes: int, samples: int, width: int, data_key: str = "hsi",
        ignored_classes: Sequence[int] = ()
    ) -> None:
        assert width % 2 == 1, "cube width must be odd"
        super().__init__(samples, width, data_key)
        self.num_classes = num_classes
        self.ignore_classes = ignored_classes

    @override
    def generate_cubes(self, data: np.ndarray, label: np.ndarray):
        slide_data = np.lib.stride_tricks.sliding_window_view(
            data, (self.width, self.width), axis=(0, 1)  # type: ignore
        )
        slide_label = np.lib.stride_tricks.sliding_window_view(
            label, (self.width, self.width), axis=(0, 1)  # type: ignore
        )
        h, w, *_ = slide_data.shape
        half_width = (self.width - 1) // 2
        mask = slide_label[:, :, half_width, half_width] != 255
        actual_mask = np.zeros_like(mask)
        # choose self.samples samples from mask
        for c in range(self.num_classes):
            if c in self.ignore_classes:
                continue
            class_mask = np.zeros_like(mask)
            label_sum = (label == c).sum()
            if label_sum == 0:
                continue
            count = 0
            while count < min(self.samples, label_sum):
                x = np.random.randint(0, h)
                y = np.random.randint(0, w)
                if mask[x, y] and not class_mask[x, y]:
                    class_mask[x, y] = True
                    count += 1
            actual_mask = actual_mask | class_mask
        out = slide_data[actual_mask]
        out_l = slide_label[actual_mask, half_width, half_width]
        return out, out_l


class GenerateHSITestCube(GenerateHSICube):
    @override
    def generate_cubes(self, data: np.ndarray, label: np.ndarray):
        half_width = (self.width - 1) // 2
        # pad data with zeros
        pad_data = np.pad(
            data,
            ((half_width, half_width), (half_width, half_width), (0, 0)),
            mode="edge",
        )
        slide_data = np.lib.stride_tricks.sliding_window_view(
            pad_data,
            (self.width, self.width),
            axis=(0, 1),  # type: ignore
        )
        return slide_data, label


class Collect:
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    def __call__(self, results):
        return {key: results[key] for key in self.keys}


if __name__ == "__main__":
    data = np.random.random((512, 512, 128))
    label = np.random.randint(0, 19, size=(512, 512), dtype=np.uint8)
    out = np.lib.stride_tricks.sliding_window_view(data, (17, 17), axis=(0, 1))  # type: ignore
    out_l = np.lib.stride_tricks.sliding_window_view(label, (17, 17), axis=(0, 1))  # type: ignore
    h, w, *_ = out.shape
    mask = out_l[:, :, 8, 8] != 255

    actual_mask = np.zeros_like(mask)
    # choose 1000 samples from mask
    count = 0
    while count < 10000:
        x = np.random.randint(0, h)
        y = np.random.randint(0, w)
        if mask[x, y] and not actual_mask[x, y]:
            actual_mask[x, y] = True
            count += 1

    masked_out = out[actual_mask]
    masked_label = out_l[actual_mask, 8, 8]
    # print(out.shape)
    # print(out_l.shape)
    print(mask.shape)
    print(masked_out.shape)
    print(masked_label.shape)
