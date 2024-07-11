import copy
import multiprocessing
import os
import pickle
import random
import cv2
from sklearn.calibration import LinearSVC
from models import TwoCNN, RSSAN
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

data_root = "data/LIB-HSI/train/"
import torch
from torch import nn

CLASSES = [
    "Vegetation Plant", "Glass Window", "Brick Wall", "Concrete Ground",
    "Block Wall", "Concrete Wall", "Concrete Footing", "Concrete Beam",
    "Brick Ground", "Glass door", "Vegetation Ground", "Soil", "Metal Sheet",
    "Woodchip Ground", "Wood/Timber Smooth Door", "Tiles Ground",
    "Pebble-Concrete Beam", "Pebble-Concrete Ground", "Pebble-Concrete Wall",
    "Metal Profiled Sheet", "Door-plastic", "Wood/timber Wall", "Wood Frame",
    "Metal Smooth Door", "Concrete Window Sill", "Metal Profiled Door",
    "Wood/Timber Profiled Door"
]

PALETTE = [[16, 122, 0], [186, 174, 69], [78, 78, 212], [228, 150, 139],
           [142, 142, 144], [222, 222, 222], [138, 149, 253], [213, 187, 160],
           [68, 229, 12], [203, 185, 82], [3, 145, 67], [100, 183, 232],
           [115, 127, 195], [16, 108, 4], [180, 186, 13], [192, 128, 255],
           [153, 31, 34], [98, 184, 210], [32, 78, 37], [99, 228, 172],
           [227, 250, 98], [150, 169, 128], [111, 182, 216], [203, 114, 188],
           [74, 5, 187], [124, 134, 113], [150, 141, 201]]


def put_palette(gray):
    im = np.zeros((*gray.shape, 3), dtype=np.uint8)
    for i, color in enumerate(PALETTE):
        im[gray == i] = color
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def train_test_split(
    x: np.ndarray,
    x_rgb: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.8,
    random_state: int | None = 42,
):
    random.seed(random_state)
    supervised = np.where(y != 255)[0].tolist()
    unsupervised = np.where(y == 255)[0].tolist()
    train_size = min(len(supervised), int(x.shape[0] * (1 - test_size)))
    random.shuffle(supervised)
    train_indices, test_indices = supervised[:train_size], supervised[
        train_size:]
    test_indices = unsupervised + test_indices
    random.shuffle(test_indices)
    x_train, x_test = x[train_indices], x[test_indices]
    x_train_rgb, x_test_rgb = x_rgb[train_indices], x_rgb[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return (x_train, x_test), (x_train_rgb,
                               x_test_rgb), (y_train, y_test), (train_indices,
                                                                test_indices)


def train_model(
    name: str,
    model: nn.Module,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    predict: np.ndarray,
    save_path: str,
    epochs: int = 10,
    ignore_index: int = 255,
):
    train_dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(
        torch.from_numpy(x_test), torch.from_numpy(y_test))
    full_dataset = TensorDataset(torch.from_numpy(predict))
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model = model.cuda()
    best_model, best_acc = {}, 0.0
    for epoch in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            if i % 50 == 0 and i != 0:
                print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.cuda()
                y = y.cuda()
                output = model(x)
                _, predicted = torch.max(output.data, 1)
                total += y[y != ignore_index].size(0)
                correct += (predicted[y != ignore_index] == y[
                    y != ignore_index]).sum().item()
        acc = 100 * correct / total
        print(f"Accuracy: {acc}")
        if acc > best_acc:
            best_acc = acc
            best_model = copy.deepcopy(model.state_dict())
            print(f"Best acc: {best_acc}")

    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, f"{name}.pt"))

    loader = DataLoader(full_dataset, batch_size=64, shuffle=False)
    model.eval()
    model.load_state_dict(best_model)
    outputs = []
    with torch.no_grad():
        for x, in loader:
            x = x.cuda()
            output = model(x)
            pred = torch.argmax(output, dim=1)
            outputs.append(pred.cpu())
    outputs = torch.cat(outputs, dim=0)
    outputs = outputs.numpy()

    return outputs


def train_svm(
    name: str,
    svm: LinearSVC,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    predict: np.ndarray,
    save_path: str,
    ignore_index: int = 255,
):
    svm.fit(x_train, y_train)
    x_test = x_test[y_test != ignore_index]
    y_test = y_test[y_test != ignore_index]
    print("Accuracy:", svm.score(x_test, y_test))

    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, f"{name}.pkl"), "wb") as f:
        pickle.dump(svm, f)

    outputs = svm.predict(predict)
    return outputs


def save_weakgt(
    name: str,
    indices: tuple[np.ndarray, np.ndarray],
    outputs: np.ndarray,
    lsc_path: str,
    save_path: str,
):
    lsc = np.load(lsc_path)
    gt = np.zeros_like(lsc, dtype=np.uint8) + 255
    for i, (x, y) in enumerate(zip(*indices)):
        gt[x, y] = outputs[i]
    im = put_palette(gt)
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, f"{name}_gray.png"), gt)
    cv2.imwrite(os.path.join(save_path, f"{name}.png"), im)


def save_outputs(
    name: str,
    indices: tuple[np.ndarray, np.ndarray],
    outputs: np.ndarray,
    lsc_path: str,
    save_path: str,
):
    lsc = np.load(lsc_path)
    label_mapping = {c: [] for c in np.unique(lsc)}

    for i, (x, y) in enumerate(zip(*indices)):
        c = outputs[i]
        k = lsc[x, y]
        label_mapping[k].append(c)

    label_mapping = {
        k: np.bincount(v).argmax()
        for k, v in label_mapping.items() if v
    }

    label = np.zeros_like(lsc, dtype=np.uint8) + 255
    for k, c in label_mapping.items():
        label[lsc == k] = c

    im = put_palette(label)
    cv2.imwrite(os.path.join(save_path, f"{name}_gray.png"), label)
    cv2.imwrite(os.path.join(save_path, f"{name}.png"), im)


def process(
    name: str,
    segment_dir: str = "slic",
    test_size: float = 4 / 5,
    random_state: int | None = 42,
):
    path = os.path.join(data_root, "samples", name + ".pt")
    data = torch.load(path)
    lsc_path = os.path.join(data_root, segment_dir, name + ".npy")
    twocnn_path = os.path.join(data_root, "generated-twocnn")
    rssan_path = os.path.join(data_root, "generated-rssan")
    svm_path = os.path.join(data_root, "generated-svm")
    rgbsvm_path: str = os.path.join(data_root, "generated-rgbsvm")
    rgbtwocnn_path: str = os.path.join(data_root, "generated-rgbtwocnn")
    weak_path = os.path.join(data_root, "sampled-gt")

    os.makedirs(twocnn_path, exist_ok=True)
    os.makedirs(rssan_path, exist_ok=True)
    os.makedirs(svm_path, exist_ok=True)
    os.makedirs(rgbsvm_path, exist_ok=True)
    os.makedirs(weak_path, exist_ok=True)
    os.makedirs(rgbtwocnn_path, exist_ok=True)

    img = data["normalized_hsi"]
    img = np.nan_to_num(img, copy=False)
    img_rgb = data["rgb"]
    gt = data["gt"]
    indices = data["indices"]

    x, x_rgb, y, (train_indices, _) = train_test_split(img, img_rgb, gt,
                                                       test_size, random_state)
    y_train, _ = y
    if np.unique(y_train).size == 1:
        return
    twocnn = TwoCNN(700, 600, 27).cuda()
    twocnnrgb = TwoCNN(20, 64, 27).cuda()
    rssan = RSSAN(175, 27).cuda()
    svm = LinearSVC(
        C=0.5,
        max_iter=40000,
        random_state=42,
        class_weight="balanced",
        verbose=0,
    )
    svmrgb = LinearSVC(
        C=0.5,
        max_iter=40000,
        random_state=42,
        class_weight="balanced",
        verbose=0,
    )

    save_weakgt(name, (indices[0][train_indices], indices[1][train_indices]),
                gt[train_indices], lsc_path, weak_path)

    print("Training SVM...")
    svm_outputs = train_svm(
        name,
        svm,
        *x,
        *y,
        img,
        svm_path,
    )
    # svm_outputs[train_indices] = gt[train_indices]
    save_outputs(name, indices, svm_outputs, lsc_path, svm_path)

    print("Training SVM(RGB)...")
    svmrgb_outputs = train_svm(
        name,
        svmrgb,
        *x_rgb,
        *y,
        img_rgb,
        rgbsvm_path,
    )
    svmrgb_outputs[train_indices] = gt[train_indices]
    save_outputs(name, indices, svmrgb_outputs, lsc_path, rgbsvm_path)

    print("Training twocnn...")
    twocnn_outputs = train_model(
        name,
        twocnn,
        *x,
        *y,
        img,
        twocnn_path,
    )
    twocnn_outputs[train_indices] = gt[train_indices]
    save_outputs(name, indices, twocnn_outputs, lsc_path, twocnn_path)
    
    print("Training twocnn rgb...")
    x1, x2 = x_rgb
    x1 = (x1 / 255).astype(np.float32)
    x2 = (x2 / 255).astype(np.float32)
    twocnnrgb_outputs = train_model(
        name,
        twocnnrgb,
        x1, x2,
        *y,
        (img_rgb / 255).astype(np.float32),
        twocnn_path,
    )
    twocnnrgb_outputs[train_indices] = gt[train_indices]
    save_outputs(name, indices, twocnnrgb_outputs, lsc_path, rgbtwocnn_path)

    # print("Training RSSAN...")
    # rssan_outputs = train_model(
    #     name,
    #     rssan,
    #     *x,
    #     *y,
    #     img,
    #     rssan_path,
    # )
    # rssan_outputs[train_indices] = gt[train_indices]
    # save_outputs(name, indices, rssan_outputs, lsc_path, rssan_path)

    print("done:", name)


names = list(
    map(
        lambda f: f[:-3],
        filter(lambda f: f.endswith(".pt"),
               os.listdir(os.path.join(data_root, "samples"))),
    ))
print(len(names))
# for name in names:
#     process(name)

pool = multiprocessing.Pool(12)
pool.map(process, names)
