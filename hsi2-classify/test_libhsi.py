import os
import torch.utils.data
import torch
from datasets.libhsi import LIBHSIDataset
from datasets.pipelines import (
    Collect,
    GenerateHSITestCube,
    LoadENVIFromFile,
    BalancedHSICubes,
    LoadHSIFromFile,
    LoadRGBFromFile,
)
from datasets import HSICityV2Dataset
from models import model_map
from functions import get_config, test, init_logger


def main():
    parser = get_config()
    parser.add_argument("--state_dict", type=str)
    args = parser.parse_args()
    test_dataset = LIBHSIDataset(
        data_root=os.path.join(args.data_root, "test"),
        pipelines=[
            LoadENVIFromFile(),
            LoadRGBFromFile(),
            GenerateHSITestCube(samples=args.samples, width=args.cube_width),
            Collect(keys=["name", "cubes", "labels"]),
        ],
    )

    logger, work_dir = init_logger("test", args.model, args.cube_width, args.work_dir)
    ModelType = model_map[args.model]
    model = ModelType(204, 27, windows=args.cube_width)

    state_dict = torch.load(args.state_dict)
    model.load_state_dict(state_dict)
    device = torch.device(args.device)
    model.to(device)

    IU_array, mIoU, acc, outputs = test(
        logger,
        model,
        test_dataset,
        device,
        num_classes=27,
        save_dir=os.path.join(work_dir, "test_results"),
    )
    logger.info(f"Test done, acc: {acc}, mIoU:{mIoU}, IU_array: {IU_array}")
    torch.save(outputs, os.path.join(work_dir, "outputs.pth"))


if __name__ == "__main__":
    main()
