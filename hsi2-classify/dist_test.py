import os
import multiprocessing as mp
from typing import Sequence
import torch.utils.data
import torch
from datasets.pipelines import (
    Collect,
    GenerateHSITestCube,
    LoadHSIFromFile,
    LoadRGBFromFile,
)
from datasets import HSICityV2Dataset
from models import model_map
from functions import get_config, test, init_logger


def worker(
    names: Sequence[str],
    data_root: str,
    cube_width: int,
    state_dict: str,
    model,
    device,
    work_dir,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    work_dir = os.path.join(work_dir, f"device-{device}")
    os.makedirs(work_dir, exist_ok=True)

    logger, work_dir = init_logger("test", model, cube_width, work_dir)

    ModelType = model_map[model]
    model = ModelType(128, 19, windows=cube_width)
    state_dict = torch.load(state_dict)
    model.load_state_dict(state_dict)
    device = torch.device("cuda")
    model.to(device)

    test_dataset = HSICityV2Dataset(
        data_root=os.path.join(data_root, "train"),
        pipelines=[
            LoadHSIFromFile(),
            LoadRGBFromFile(),
            GenerateHSITestCube(samples=10000, width=cube_width),
            Collect(keys=["cubes", "labels", "name"]),
        ],
        name_list=names,
    )
    IU_array, mIoU, acc, outputs = test(
        logger,
        model,
        test_dataset,
        device,
        num_classes=19,
        save_dir=os.path.join(work_dir, "test_results"),
    )
    logger.info(f"Test done, acc: {acc}, mIoU:{mIoU}, IU_array: {IU_array}")
    torch.save(outputs, os.path.join(work_dir, "outputs.pth"))


def main():
    parser = get_config()
    parser.add_argument("--state_dict", type=str)
    args = parser.parse_args()
    with open("names.lst", "r") as f:
        names = [name.strip() for name in f.readlines()]
    # split names into 4 parts
    names = [names[i::8] for i in range(8)]
    print(names)

    procs = [
        mp.Process(
            target=worker,
            args=(
                names[i],
                args.data_root,
                args.cube_width,
                args.state_dict,
                args.model,
                i % 4,
                args.work_dir,
            ),
        )
        for i in range(8)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()
