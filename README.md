# Multitask Perception for Autonomous Driving

Unified PyTorch framework for multitask perception combining:
- **Object Detection** (NanoDet, CenterNet)
- **Semantic Segmentation** (SegFormer-B0, DeepLabV3, ESPNetV2)
- **Monocular Depth Estimation** (Planned)

Built with Python 3.12, PyTorch 2.2, and designed for efficient training on resource-constrained GPUs.

## Features

- Efficient multitask learning with shared backbone
- Modular architecture - easy to extend with new heads/backbones
- Multiple backbone options from lightweight to high-accuracy
- Configurable via YAML experiment files
- Docker support for reproducibility

## Requirements

- Python 3.12+
- CUDA 11.8+
- 16GB+ RAM

## Installation

### Poetry (Recommended)

```bash
git clone https://github.com/ratnajitmukherjee/multitask-perception.git
cd multitask-perception

poetry install
poetry shell

python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Docker

```bash
docker build -t multitask-perception:latest .
docker run --gpus all -it multitask-perception:latest
```

## Quick Start

### Training

```bash
# CenterNet detection on VOC
python train.py --config-file configs/detection/centernet/hardnet68_cnet512_voc0712.yaml

# NanoDet multitask (detection + segmentation)
python train.py --config-file configs/multitask/deeplabv3/multitask_vovnet39_deeplabv3_vps_512.yaml
```

### Evaluation

```bash
python test.py --config-file configs/detection/centernet/hardnet68_cnet512_voc0712.yaml \
    --ckpt outputs/model_final.pth
```

### Inference

```bash
python infer.py --config-file configs/detection/centernet/hardnet68_cnet512_voc0712.yaml \
    --ckpt outputs/model_final.pth \
    --input path/to/images/ \
    --output_dir outputs/predictions/
```

## Supported Backbones

| Backbone | Params | Best For |
|----------|--------|----------|
| **MobileNetV3-Small** | 2.5M | Edge devices |
| **MobileNetV3-Large** | 5.4M | Mobile |
| **VoVNet-27-slim** | 3.5M | Real-time |
| **VoVNet-39** | 22.6M | Balanced |
| **HarDNet-68** | 17.6M | Efficient |
| **HarDNet-85** | 36.7M | High accuracy |

## Supported Heads

### Detection
- **NanoDet**: Anchor-free, lightweight, fast
- **CenterNet**: Keypoint-based detection (optional DCN via `USE_DCN` config flag)

### Segmentation
- **SegFormer-B0**: Transformer-based
- **DeepLabV3**: Atrous convolution based
- **ESPNetV2**: Ultra-efficient

### Depth (Planned)
- Simple decoder architecture

## Project Structure

```
multitask-perception/
├── train.py                    # Training entry point
├── test.py                     # Evaluation entry point
├── infer.py                    # Inference entry point
├── configs/                    # YAML experiment configs
│   ├── detection/              # Detection-only configs (CenterNet, NanoDet, SSD)
│   ├── segmentation/           # Segmentation-only configs (DeepLabV3, ESPNetV2)
│   └── multitask/              # Multitask configs (detection + segmentation)
├── src/multitask_perception/
│   ├── config/                 # Python defaults and head sub-configs
│   ├── data/                   # Datasets, transforms, samplers
│   ├── modeling/
│   │   ├── backbones/          # HarDNet, VoVNet, MobileNetV3
│   │   ├── heads/
│   │   │   ├── detection/      # CenterNet, NanoDet
│   │   │   ├── segmentation/   # SegFormer, DeepLabV3, ESPNetV2
│   │   │   └── depth/          # Planned
│   │   ├── layers/             # SeparableConv2d, L2Norm, EfficientPWConv
│   │   └── losses/             # FocalLoss, SegmentationLoss
│   ├── engine/                 # Training and evaluation loops
│   ├── solver/                 # Optimizers and LR schedulers
│   ├── structures/             # Container for detection outputs
│   └── utils/                  # Checkpointing, metrics, visualization
├── scripts/                    # Dataset preparation scripts
├── pyproject.toml              # Poetry project config
├── Dockerfile
└── docker-compose.yml
```

## Configuration

Python defaults live in `src/multitask_perception/config/defaults.py`. Experiment-specific overrides are YAML files under `configs/`:

```yaml
MODEL:
  NUM_CLASSES: 20
  BACKBONE:
    NAME: 'HarDNet68'
    OUT_CHANNEL: 1024
  HEAD:
    DET_NAME: 'CenterNetHead'
    HEAD_CONFIG: {'hm':20, 'wh': 2, 'reg': 2}
    LOSS_WEIGHTS: {'hm':1, 'wh': 0.1, 'reg': 1}
SOLVER:
  NAME: 'SGD_optimizer'
  MAX_ITER: 120000
  BATCH_SIZE: 8
  LR: 1e-3
OUTPUT_DIR: './outputs/my_experiment'
```

## Roadmap

- [x] Core multitask architecture
- [x] Detection heads (NanoDet, CenterNet)
- [x] Segmentation heads (SegFormer, DeepLabV3, ESPNetV2)
- [x] Solver module (SGD, Adam, Cosine/MultiStep/Polynomial schedulers)
- [x] Custom layers (SeparableConv2d, EfficientPyrPool, ESPNetV2 blocks)
- [ ] Depth estimation head
- [ ] Video support (temporal modules)
- [ ] Object tracking integration
- [ ] ONNX export support

## Contact

For questions or issues, please open a GitHub issue or contact ratnajitmukherjee@gmail.com
