import argparse
import os
import time
import cv2
import numpy as np
import torch.utils.data
import torch
import torch.nn as nn
import logging

from tqdm import tqdm

from datasets.hsicity2 import HSICityV2Dataset


def metrics(cm: torch.Tensor):
    """
    Calculate metrics from confusion matrix
    :param cm: confusion matrix
    :return: metrics
    """
    pos = cm.sum(1)
    res = cm.sum(0)
    tp = torch.diag(cm)
    # IoU
    IU_array = (tp / (pos + res - tp + 1e-10)).numpy()
    mIoU = IU_array.mean()
    # Pixel Accuracy
    acc = tp.sum() / cm.sum()
    return IU_array, mIoU, acc.item()


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="data/LIB-HSI")
   
    parser.add_argument("--cube_bs", type=int, default=256)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model", type=str, default="RSSAN")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--cube_width", type=int, default=17)
    parser.add_argument("--work_dir", type=str, default="libhsi_work_dirs")

    return parser


def init_logger(name: str, model_name: str, cube_width: int, work_dir: str):
    current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    log_path = os.path.join(work_dir, f"{model_name}-w{cube_width}")
    os.makedirs(log_path, exist_ok=True)
    log_file = os.path.join(log_path, f"{current_time}.log")

    file = logging.FileHandler(log_file)
    console = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s %(message)s")
    file.setFormatter(formatter)
    console.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.addHandler(file)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)

    return logger, log_path


def train(
    logger: logging.Logger,
    epoch: int,
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ExponentialLR,
    loss_fn: nn.Module,
    device: torch.device,
    cube_bs: int = 256,
    log_interval: int = 10,
):
    model.train()
    losses = 0
    tic = time.time()
    current_lr = optimizer.param_groups[0]["lr"]
    for i, batch in enumerate(tqdm(loader)):
        cubes, labels = batch["cubes"], batch["labels"]
        b, s, c, h, w = cubes.shape
        cubes = cubes.reshape(b * s, c, h, w)
        labels = labels.reshape(b * s)
        cube_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(cubes, labels),
            batch_size=cube_bs,
            shuffle=True,
        )
        for cube_batch in cube_loader:
            cubes, labels = cube_batch
            cubes = cubes.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(cubes)
            loss = loss_fn(logits, labels)
            loss.backward()
            losses += loss.item()
            optimizer.step()
            print(f"loss:{loss.item():.4f}", end="\r")
        if i % log_interval == 0 or i == len(loader) - 1:
            avg_loss = losses / cube_bs / (i + 1)
            cost_time = (time.time() - tic) // 60
            logger.info(
                f"[{i}/{len(loader)}] epoch:{epoch}, lr:{current_lr:.4f}, avg_loss:{avg_loss:.4f}, cost_time:{cost_time}m"
            )

    scheduler.step()
    toc = time.time()
    loss = losses / len(loader) / cube_bs
    return model, toc - tic, loss


class TestCube(torch.utils.data.Dataset):
    # due to the memory limit, we need to test the model in a cube-by-cube manner
    def __init__(self, cubes, labels) -> None:
        self.cubes = cubes
        self.labels = labels

        h, w, *_ = cubes.shape
        self.img_shape = (h, w)
    
    def __len__(self) -> int:
        h, w = self.img_shape
        return h
    
    def __getitem__(self, i: int):
        cubes = torch.from_numpy(np.ascontiguousarray(self.cubes[i, ...].copy()))
        labels = self.labels[i, ...]
        return cubes, labels



def test(
    logger: logging.Logger,
    model: nn.Module,
    dataset: HSICityV2Dataset,
    device: torch.device,
    num_classes: int = 19,
    save_dir: str = "work_dirs",
):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    total_confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    results = []
    for e in range(len(dataset)):
        info = dataset[e]
        confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
        cubes, labels, name = info["cubes"], info["labels"], info["name"]
        if os.path.exists(os.path.join(save_dir, f"rgb{name}.png")):
            continue
        cube_loader = torch.utils.data.DataLoader(
            TestCube(cubes, labels),
            batch_size=1,
            shuffle=False,
        )
        outputs = []

        with torch.no_grad():
            for i, cube_batch in enumerate(cube_loader):
                cubes, labels = cube_batch
                cubes = cubes.squeeze(0)
                labels = labels.squeeze(0).numpy()
                cubes = cubes.to(device)
                logits = model(cubes)
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                for t, p in zip(labels.reshape(-1), pred.reshape(-1)):
                    t_class = t.astype(np.int64)
                    p_class = p.astype(np.int64)
                    if t_class == 255:
                        continue
                    confusion_matrix[t_class, p_class] += 1
                print(pred)
                outputs.append(pred)
                print(f"[{i}/{len(cube_loader)}]", end="\r")
        total_confusion_matrix += confusion_matrix
        output_label = np.stack(outputs, axis=0)
        rgb_label = dataset.put_palette(output_label)
        cv2.imwrite(os.path.join(save_dir, f"rgb{name}.png"), rgb_label)
        cv2.imwrite(os.path.join(save_dir, f"rgb{name}_gray.png"), output_label)
        IU_array, mIoU, acc = metrics(confusion_matrix)
        logger.info(
            f"[{e}/{len(dataset)}] name: {name}, acc:{acc:.4f}, mIoU: {mIoU:.4f}, mIoU_array:{IU_array}"
        )
        result = (name, output_label, confusion_matrix.numpy())
        results.append(result)

    IU_array, mIoU, acc = metrics(total_confusion_matrix)
    return IU_array, mIoU, acc, results
