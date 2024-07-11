from collections import OrderedDict
import os
from typing import Sequence
import cv2
import torch.nn.functional as F
import numpy as np
import torch


def intersect_and_union(
    pred: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
    ignore_index: int,
):
    mask = label != ignore_index
    pred = pred[mask]
    label = label[mask]

    intersect = pred[pred == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1
    )
    area_pred_label = torch.histc(
        pred.float(), bins=(num_classes), min=0, max=num_classes - 1
    )
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1
    )
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(
    preds: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    num_classes: int,
    ignore_index: int,
):
    total_area_intersect = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes,), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes,), dtype=torch.float64)
    for result, gt_seg_map in zip(preds, labels):
        area_intersect, area_union, area_pred_label, area_label = intersect_and_union(
            result, gt_seg_map, num_classes, ignore_index
        )
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return (
        total_area_intersect,
        total_area_union,
        total_area_pred_label,
        total_area_label,
    )


def calculate_metrics(
    total_area_intersect,
    total_area_union,
    total_area_pred_label,
    total_area_label,
    nan_to_num=None,
):
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({"aAcc": all_acc})

    iou = total_area_intersect / total_area_union
    acc = total_area_intersect / total_area_label
    ret_metrics["IoU"] = iou
    ret_metrics["Acc"] = acc

    ret_metrics = {metric: value.numpy() for metric, value in ret_metrics.items()}
    if nan_to_num is not None:
        ret_metrics = OrderedDict(
            {
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            }
        )
    return ret_metrics


def mean_iou(
    preds: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    num_classes: int,
    ignore_index: int,
    nan_to_num: int | None = None,
):
    metrics = calculate_metrics(
        *total_intersect_and_union(preds, labels, num_classes, ignore_index),
        nan_to_num,
    )

    return metrics


def format_metrics(classes: Sequence[str], metrics: dict[str, np.ndarray]):
    all_acc = metrics["aAcc"] * 100
    iou = metrics["IoU"] * 100
    acc = metrics["Acc"] * 100

    class_table = ["Per class IoU, Acc:"]
    maxlen = max(len(class_name) for class_name in classes)

    for class_name, class_iou, class_acc in zip(classes, iou, acc):
        class_table.append(f"{class_name:<{maxlen}}: {class_iou:.3f}, {class_acc:.3f}")

    mIoU = np.nanmean(iou)
    mAcc = np.nanmean(acc)

    metric_str = f"Evaluation: mIoU: {mIoU:.3f}, mAcc: {mAcc:.3f}, aAcc: {all_acc:.3f}"
    class_metric_str = "\n".join(class_table)
    return metric_str, class_metric_str


def evaluate(
    classes: Sequence[str],
    preds: Sequence[torch.Tensor],
    labels: Sequence[torch.Tensor],
    method: str = "mIoU",
    ignore_index: int = 255,
) -> tuple[str, str]:
    assert method == "mIoU", "default evaluation only supports mIoU now"
    num_classes = len(classes)
    metrics = mean_iou(preds, labels, num_classes, ignore_index)
    metric_str, class_metric_str = format_metrics(classes, metrics)
    return metric_str, class_metric_str


allowed = ["JigSawHSI", "RSSAN"]
work_dir = "libhsi_work_dirs"
# rgb_work_dir = "/data1/iori/work_dirs_rgb"
gt_dir = "/data1/iori/LIB-HSI/test/labels"

# CLASSES = [
#     "road",
#     "sidewalk",
#     "building",
#     "wall",
#     "fence",
#     "pole",
#     "traffic light",
#     "traffic sign",
#     "vegetation",
#     "terrain",
#     "sky",
#     "person",
#     "rider",
#     "car",
#     "truck",
#     "bus",
#     "train",
#     "motorcycle",
#     "bicycle",
# ]
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

def evaluate_experiment(exp_name: str, path: str):
    print("Experiment:", exp_name)
    preds = []
    gts = []
    for file in os.listdir(path):
        if not file.endswith("_gray.png"):
            continue
        pred = cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(os.path.join(gt_dir, file[3:]), cv2.IMREAD_GRAYSCALE)
        if gt is None:
            continue
        h, w = pred.shape
        gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
        preds.append(torch.from_numpy(pred))
        gts.append(torch.from_numpy(gt))
    metric_str, class_metric_str = evaluate(
        classes=CLASSES, preds=preds, labels=gts, method="mIoU"
    )
    print(metric_str)
    print(class_metric_str)


for entry in os.listdir(work_dir):
    if any(entry.startswith(allow) for allow in allowed):
        evaluate_experiment(entry, os.path.join(work_dir, entry, "test_results"))
# for entry in os.listdir(rgb_work_dir):
#     if any(entry.startswith(allow) for allow in allowed):
#         evaluate_experiment(entry + "_rgb", os.path.join(rgb_work_dir, entry, "test_results"))
# evaluate_experiment("svm", "/home/tianqi/workspace/hsi2-classify/svm/hsi_svm_out")
# evaluate_experiment("svm_rgb", "/home/tianqi/workspace/hsi2-classify/svm/rgb_svm_out")