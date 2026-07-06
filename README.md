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
