# Crack Segmentation using U-Net

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

![Project Banner](docs/images/banner.webp)
A deep learning project for automatic crack detection and segmentation using a lightweight U-Net architecture.
```mermaid
graph LR
A[📷 Input Image] --> B[🧠 U-Net Model] --> C[🎭 Segmentation Mask]
```
## Overview

This project implements semantic segmentation for detecting cracks in infrastructure images. Built with PyTorch, it uses a simplified U-Net with skip connections, trained on COCO-formatted datasets.

## Architecture

```mermaid
flowchart LR
subgraph Encoder
E1[32] --> E2[64] --> E3[128]
end

E3 --> B[256]

subgraph Decoder
B --> D3[128] --> D2[64] --> D1[32]
end

E1 -.-> D1
E2 -.-> D2
E3 -.-> D3
```
## Quick Start

bash
# Clone and install
git clone https://github.com/ghobadian/asphalt_crack_segmentation.git
cd crack-segmentation-unet
pip install -r requirements.txt

# Run the full pipeline
python src/pipeline.py


## Dataset

Expects COCO-format annotations. Place your data in `data/raw/` with `train/`, `valid/`, `test/` subfolders, each containing images and `_annotations.coco.json`.

## Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 512 × 512 |
| Batch Size | 14 |
| Epochs | 50 |
| Learning Rate | 1e-4 |

## Results

The model outputs binary segmentation masks and is evaluated using IoU (Intersection over Union).

## License

MIT License 


