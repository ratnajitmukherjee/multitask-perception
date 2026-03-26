# MULTITASK PERCEPTION FOR AUTONOMOUS DRIVING
Unified PyTorch framework for multitask perception combining:
- **Object Detection** (NanoDet, CenterNet)
- **Semantic Segmentation** (SegFormer-B0, DeepLabV3, ESPNetV2)
- **Monocular Depth Estimation** (Coming soon)

Built with Python 3.12, PyTorch 2.2, and designed for efficient training on resource-constrained GPUs.

## 🚀 Features

- ✅ Efficient multitask learning with shared backbone
- ✅ Support for multiple datasets (KITTI, nuScenes, BDD100K, Cityscapes)
- ✅ Pseudo-labeling pipeline for depth estimation
- ✅ Modular architecture - easy to extend
- ✅ Video-ready design (temporal modules planned)
- ✅ Docker support for reproducibility
- ✅ Comprehensive type hints and documentation

## 📋 Requirements

- Python 3.12+
- CUDA 11.8+ (tested on GTX 1080 Ti)
- 16GB+ RAM
- 50GB+ storage for datasets

## 🔧 Installation

### Option 1: Poetry (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/multitask-perception.git
cd multitask-perception

# Install with Poetry
poetry install

# Activate environment
poetry shell

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### Option 2: pip

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e .
```

### Option 3: Docker

```bash
# Build image
docker build -t multitask-perception:latest .

# Run container
docker run --gpus all -it multitask-perception:latest
```

## 🎯 Quick Start

### Training

```bash
# Detection only
python src/multitask_perception/tools/train.py \
    --config configs/tasks/detection_only.yaml

# Segmentation only
python src/multitask_perception/tools/train.py \
    --config configs/tasks/segmentation_only.yaml

# Full multitask (detection + segmentation)
python src/multitask_perception/tools/train.py \
    --config configs/experiments/kitti_multitask.yaml
```

### Evaluation

```bash
python src/multitask_perception/tools/eval.py \
    --config configs/experiments/kitti_multitask.yaml \
    --checkpoint outputs/checkpoints/best_model.pth
```

### Inference

```bash
python src/multitask_perception/tools/infer.py \
    --config configs/experiments/kitti_multitask.yaml \
    --checkpoint outputs/checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --output outputs/predictions/
```

## 📊 Supported Backbones

| Backbone | Params | Speed | Best For |
|----------|--------|-------|----------|
| **MobileNetV3-Small** | 2.5M | ⚡⚡⚡ | Edge devices |
| **MobileNetV3-Large** | 5.4M | ⚡⚡ | Mobile |
| **VoVNet-27-slim** | 3.5M | ⚡⚡⚡ | Real-time |
| **VoVNet-39** | 22.6M | ⚡⚡ | Balanced |
| **HarDNet-68** | 17.6M | ⚡⚡ | Efficient |
| **HarDNet-85** | 36.7M | ⚡ | High accuracy |

## 🎯 Supported Heads

### Detection
- **NanoDet**: Anchor-free, fast, accurate
- **CenterNet**: Keypoint-based detection

### Segmentation
- **SegFormer-B0**: Modern transformer, 3.7M params
- **DeepLabV3**: Classic, high accuracy, 40M+ params
- **ESPNetV2**: Ultra-efficient, 0.8M params

### Depth (Coming Soon)
- Simple decoder
- MonoDepth-style

## 📁 Project Structure

```
multitask-perception/
├── src/multitask_perception/
│   ├── core/
│   │   ├── config/          # Configuration system
│   │   ├── data/            # Datasets and transforms
│   │   ├── modeling/        # Models, backbones, heads
│   │   ├── engine/          # Training and evaluation
│   │   ├── solver/          # Optimizers and schedulers
│   │   └── utils/           # Utilities
│   └── tools/               # Training/eval scripts
├── configs/                 # Configuration files
├── scripts/                 # Helper scripts
├── tests/                   # Unit tests
└── docs/                    # Documentation
```

## 🔬 Dataset Preparation

### KITTI

```bash
# Download KITTI dataset
# See docs/dataset_preparation.md for details

python scripts/dataset_preparation/prepare_kitti.py \
    --data-root data/kitti \
    --output data/kitti_processed
```

### nuScenes

```bash
python scripts/dataset_preparation/prepare_nuscenes.py \
    --data-root data/nuscenes \
    --output data/nuscenes_processed
```

## 📝 Configuration

Configurations are in YAML format:

```yaml
# configs/experiments/my_experiment.yaml
TASK:
  ENABLED: ['detection', 'segmentation']

MODEL:
  BACKBONE:
    NAME: 'vovnet39'
  HEADS:
    DETECTION:
      NAME: 'NanoDet'
    SEGMENTATION:
      NAME: 'SegFormer'

SOLVER:
  BASE_LR: 0.001
  MAX_ITER: 100000
```

## 🐳 Docker

```bash
# Build
docker build -t multitask-perception:latest .

# Run training
docker-compose up train

# Run Jupyter
docker-compose up jupyter

# Interactive shell
docker run --gpus all -it multitask-perception:latest bash
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_modeling/test_model.py

# With coverage
pytest --cov=src/multitask_perception
```

## 📈 Logging

### TensorBoard

```bash
tensorboard --logdir outputs/logs
```

### Weights & Biases

```bash
# Set API key
export WANDB_API_KEY=your_key

# Training will auto-log to W&B
```

## 🚧 Roadmap

- [x] Core multitask architecture
- [x] Detection heads (NanoDet, CenterNet)
- [x] Segmentation heads (SegFormer, DeepLabV3, ESPNetV2)
- [ ] Depth estimation head
- [ ] Pseudo-labeling pipeline
- [ ] Video support (temporal modules)
- [ ] Object tracking integration
- [ ] Model optimization (ONNX, TensorRT)
- [ ] Deployment tools

## 📚 Documentation

See [docs/](docs/) for detailed documentation:
- [Installation Guide](docs/installation.md)
- [Dataset Preparation](docs/dataset_preparation.md)
- [Training Guide](docs/training.md)
- [Configuration Reference](docs/configuration.md)

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- Based on research in multitask learning for autonomous driving
- Inspired by MMDetection, Detectron2, and SegFormer architectures
- Built with PyTorch and modern ML best practices

## 📧 Contact

For questions or issues, please open a GitHub issue or contact [ratnajitmukherjee@gmail.com]

---

**Happy Training!** 🚀
