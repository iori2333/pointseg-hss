# PointSeg-HSS

> Code implementation of CVPRW2024 *Point-Supervised Semantic Segmentation of Natural Scenes via Hyperspectral Imaging*.
> 
> Material download: [OneDrive](https://smailnjueducn-my.sharepoint.com/:f:/g/personal/tianqi_ren_smail_nju_edu_cn/EjrYjrlGXiRMm8m07UiSBt8B8xKEhkTqaaYsS3F7G2vhXQ?e=Cay0Oo)

## Preparation

### MMSegmentation

Our code implements the method based on MMSegmentation. Therefore, you need to install it first. Switch to `mmsegmentation`, and run `pip install -e .` to install it. We recommend using `Python 3.10` with `PyTorch 1.11` via `miniconda`.

For more information about `mmsegmentation`, please refer to [MMSegmentation Docs](https://mmsegmentation.readthedocs.io/zh-cn/latest/).

### Datasets

Since our method is based on pointly-annotated datasets, which is not common in semantic segmentation, we provide preprocessed datasets for training and evaluation of our method. Our experiments are conducted on `HSICityV2` and `LIB-HSI`:

|  Dataset  | Train |  Val  | Test  | Num classes |
| :-------: | :---: | :---: | :---: | :---------: |
| HSICityV2 | 1030  |  108  |  276  |     19      |
|  LIB-HSI  |  393  |  46   |  76   |     27      |

> Note that we ignore serval categories in `HSICityV2` and `LIB-HSI`, since these categories contains whiteboard, camera and other objects and are mainly used for hyperspectral validation and correction.

We sampled points, as well as their coordinates and labels, from each hyperspectral image, forming an image dict in `.pt` format with following keys:
- `normalized_hsi`: a list of normalized hyperspectral points.
- `rgb`: a list of RGB points with same coordinates as `normalized_hsi`, which are used for visualization and performance comparison
- `gt`: a list of ground truth labels with same coordinates as `normalized_hsi`. In experiments, only few of them are included for training.
- `indices`: the coordinates of points.

To view a single hsi-dict, use following code:
```python
path = os.path.join(data_root, "samples", name + ".pt")
data = torch.load(path)
"""
{
  "normalized_hsi": [...],
  "rgb": [...],
  "gt": [...],
  "indices": [...]
}
"""
```

Dataset files are be downloaded from [OneDrive](https://smailnjueducn-my.sharepoint.com/:f:/g/personal/tianqi_ren_smail_nju_edu_cn/EjrYjrlGXiRMm8m07UiSBt8B8xKEhkTqaaYsS3F7G2vhXQ?e=Cay0Oo).

## Experiments

> Experiments of our method are placed under `mmsegmentation` folder, so chdir into `mmsegmentation` before running.

### Annotation Generation

Make sure you have downloaded the datasets and placed them under `data/` folder. Code implementation of this module is under `tools/psuedo_gen`. Run:
```python
python tools/psuedo_gen/<hsicity2.py/libhsi.py>
```

This will load the dataset and generate pseudo-annotations. For validation, you can fetch download our generated pseudo-annotations from [OneDrive](https://smailnjueducn-my.sharepoint.com/:f:/g/personal/tianqi_ren_smail_nju_edu_cn/EjrYjrlGXiRMm8m07UiSBt8B8xKEhkTqaaYsS3F7G2vhXQ?e=Cay0Oo). After downloading `generated-annotations.zip`, you can extract and place them under training folders of both datasets.

### Semantic Segmentation

All training experiment configurations are stored under `experiments/` folder. To train a model, run following command:

```bash
bash tools/dist_train.sh <path/to/config> <num_gpus>
```

This will automatically download pretrained models and generates checkpoints under `work_dirs/<config_name>`. Evaluation are conducted automatically after each checkpoint, and results are included in the log file.

To conduct inference, run following command:
```bash
python tools/test.py <path/to/config> <path/to/checkpoint> [command_args...]
```
You can add more arguments to `test.py` to control the inference process, for example, draw the predicted results on the image. For more information, please refer to [MMSegmentation Docs](https://mmsegmentation.readthedocs.io/zh-cn/latest/).

### HSI2-Classify

In Sec. Experiment we compared our method with traditional classification methods. To reproduce these results, you can use code under `hsi2-classify`, which provides a simple framework for training and evaluation of HSI classification models.

To train a model on LIB-HSI for example, run following command:
```python
python train_libhsi.py --model <model> --cube_width <int> --data_root <path/to/libhsi> --work_dir <path/to/work_dir>
```

For testing, run following command:
```python
python test_libhsi.py --model <model> --cube_width <int> --state_dict <path/to/checkpoint> --data_root <path/to/libhsi> --work_dir <path/to/work_dir>
```

## Citation

```bibtex
@InProceedings{Ren_2024_CVPR,
    author    = {Ren, Tianqi and Shen, Qiu and Fu, Ying and You, Shaodi},
    title     = {Point-Supervised Semantic Segmentation of Natural Scenes via Hyperspectral Imaging},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2024},
    pages     = {1357-1367}
}
```
