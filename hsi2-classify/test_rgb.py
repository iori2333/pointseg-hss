import os
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


def main():
    parser = get_config()
    parser.add_argument("--state_dict", type=str)
    args = parser.parse_args()
    test_dataset = HSICityV2Dataset(
        data_root=os.path.join(args.data_root, "test"),
        pipelines=[
            LoadRGBFromFile(),
            GenerateHSITestCube(data_key="img", samples=args.samples, width=args.cube_width),
            Collect(keys=["cubes", "labels", "name"]),
        ],
    )

    logger, work_dir = init_logger("test", args.model, args.cube_width, args.work_dir + "_rgb")
    ModelType = model_map[args.model]
    model = ModelType(3, 19, windows=args.cube_width)
    
    state_dict = torch.load(args.state_dict)
    model.load_state_dict(state_dict)
    device = torch.device(args.device)
    model.to(device)

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


if __name__ == "__main__":
    main()
