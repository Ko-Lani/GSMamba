# GSMamba
Official code repository of 'Gather-Scatter Mamba: Accelerating Propagation with Efficient State Space Model'

## Environment Setup

```bash
conda create -n gsmamba python=3.9
conda activate gsmamba
```

### Install PyTorch

Pick the command matching your CUDA toolkit.

```bash
# cu124
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# cu118
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118

# cu128
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu128
```

### Build `causal-conv1d` and `mamba`

If you are on a GPU architecture not yet covered by the published wheels (e.g. Hopper/Blackwell, sm_90 / sm_120), add the corresponding `-gencode` flags before building from source:

```python
if bare_metal_version >= Version("11.8"):
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_90,code=sm_90")
if bare_metal_version >= Version("12.0"):
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_120,code=sm_120")
```

Then build both packages from source:

```bash
pip install ninja==1.11.1.1
pip install packaging

cd causal-conv1d
python setup.py install

cd ../mamba
python setup.py install
```

### Remaining dependencies

```bash
pip install opencv-python timm tensorboard wandb matplotlib
```

## Dataset Preparation

### REDS

Download REDS via [Git LFS](https://huggingface.co/datasets/ubin108/REDS) from Hugging Face:

```bash
apt update
apt install git-lfs -y
git lfs install

GIT_LFS_PROGRESS=1 git clone https://huggingface.co/datasets/ubin108/REDS
mv REDS Dataset
cd Dataset
git lfs pull
```

## Training

Training is driven by a YAML config under [options/](options/) and launched through [train_gsmamba.py](train_gsmamba.py). Update the `dataroot_gt` / `dataroot_lq` / `meta_info_file` paths in the config to point at your local copy of REDS before starting.

Single GPU:

```bash
python train_gsmamba.py -opt options/train_gsmamba_reds.yml
```

Multi-GPU (distributed, matches the official 8-GPU recipe used for `pretrained/gsmamba_reds_x4.pth`):

```bash
torchrun --nproc_per_node=8 --master_port=4321 train_gsmamba.py -opt options/train_gsmamba_reds.yml --launcher pytorch
```

The ablation configs ([options/ablation_k3_reds.yml](options/ablation_k3_reds.yml), [options/ablation_k5_reds.yml](options/ablation_k5_reds.yml)) are launched the same way, just swap `-opt`. Logs, checkpoints, and training states are written to `experiments/<name>/`, TensorBoard logs to `tb_logger/<name>/`.

## Testing

[test_gsmamba_reds.py](test_gsmamba_reds.py) evaluates a checkpoint on REDS4 and reports PSNR/SSIM:

```bash
python test_gsmamba_reds.py \
    --model_path pretrained/gsmamba_reds_x4.pth \
    --folder_lq /path/to/REDS4/sharp_bicubic \
    --folder_gt /path/to/REDS4/GT \
    --save_result
```

`--folder_lq` / `--folder_gt` should point at the REDS4 validation split inside the dataset prepared above (`sharp_bicubic` for the LQ input, `GT` for the ground truth). Drop `--save_result` if you only want the metrics and not the output frames (written to `results/<task>/`). The baseline models included in this repo can be evaluated the same way with their respective scripts: [test_basicvsr_reds.py](test_basicvsr_reds.py), [test_basicvsrpp_reds.py](test_basicvsrpp_reds.py), [test_vrt_reds.py](test_vrt_reds.py), [test_rvrt_reds.py](test_rvrt_reds.py), [test_iart_reds.py](test_iart_reds.py).

## Acknowledgements

This codebase builds on [BasicSR](https://github.com/XPixelGroup/BasicSR), and its baseline/component implementations are adapted from [BasicVSR](https://github.com/ckkelvinchan/BasicVSR-IconVSR), [BasicVSR++](https://github.com/ckkelvinchan/BasicVSR_PlusPlus), [VRT](https://github.com/JingyunLiang/VRT), [RVRT](https://github.com/JingyunLiang/RVRT), [IART](https://github.com/kai422/IART), [SPyNet](https://github.com/sniklaus/pytorch-spynet), and [Mamba](https://github.com/state-spaces/mamba) / [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d). We also thank the [REDS](https://seungjunnah.github.io/Datasets/reds.html) dataset authors. Many thanks to the authors of these excellent works for making their code and models publicly available.
