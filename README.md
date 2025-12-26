# Deep Learning Architectures: From-Scratch Implementations from Research Papers

This repository provides **from-scratch implementations** of influential deep learning architectures, built while studying seminal research papers in computer vision and deep learning. The emphasis is on **learning by implementation**, ensuring a deep understanding of the core concepts without relying on high-level libraries or pre-built wrappers.

All models are implemented in pure PyTorch, staying faithful to the original papers while prioritizing clean, modular, and extensible code.

---

## 🚀 Design Philosophy

The project adheres to the following principles:

- **From-scratch implementations**: Core logic (layers, blocks, etc.) is built ground-up with minimal external abstractions.
- **Readable and modular code**: Clear structure for easy understanding, debugging, and modification.
- **Paper-faithful architectures**: Strict adherence to the specifications in the original papers.
- **Scalable and extensible**: Easy adaptation to different datasets, tasks, and hardware.
- **Separation of concerns**: Dedicated modules for data handling, model definition, training logic, and utilities.

---

## 📂 Repository Structure

Each architecture has its own isolated folder for clarity and independent experimentation.

```text
├── efficientnet/
├── resnet/
├── unet/
├── unet_plus_plus/
└── ... (more coming soon)
```


### Standard Folder Layout (per architecture)..Goto to src/

| File                  | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `model.py`            | Core neural network architecture implemented from scratch                    |
| `training_pipeline.py`| Training loop, validation, loss computation, and checkpointing             |
| `data_pipeline.py`    | Dataset loading, preprocessing, augmentations, and DataLoaders              |
| `config.py`           | Hyperparameters and experiment configuration                                |
| `main.py`             | Entry point for training and inference                                      |
| `utils.py`            | Helper functions (metrics, visualization, etc.)                             |
| `logger.py`           | Logging and experiment tracking utilities                                   |
| `exception.py`        | Custom exceptions for better error handling                                 |

---

## 🏗️ Implemented Architectures

- **EfficientNet**: Compound scaling of depth, width, and resolution for efficient CNNs.
- **ResNet**: Deep residual learning with skip connections to enable very deep networks.
- **UNet**: Encoder-decoder architecture with skip connections for precise segmentation.
- **UNet++**: Nested dense skip pathways for improved segmentation accuracy.

---

## 📚 References & Citations

This project is inspired by the following foundational papers:

- **EfficientNet**: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"  
  Mingxing Tan, Quoc V. Le | ICML 2019

- **ResNet**: "Deep Residual Learning for Image Recognition"  
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun | CVPR 2016

- **UNet**: "U-Net: Convolutional Networks for Biomedical Image Segmentation"  
  Olaf Ronneberger, Philipp Fischer, Thomas Brox | MICCAI 2015

- **UNet++**: "UNet++: A Nested U-Net Architecture for Medical Image Segmentation"  
  Zongwei Zhou, Md Mahfuzur Rahman Siddiquee, Nima Tajbakhsh, Jianming Liang | DLMIA 2018 (held with MICCAI)

---

## 🛠️ Future Plans

- [ ] Benchmarking and performance comparison scripts
- [ ] Mixed-precision training (FP16 / BF16)
- [ ] Integration with experiment tracking tools (e.g., Weights & Biases)
- [ ] Transformer-based models (ViT, Swin Transformer, etc.)
- [ ] Pre-trained weights release
- [ ] Additional architectures (DenseNet, MobileNet, etc.)

---

## 🏁 Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/paper-architectures-from-scratch.git
   cd paper-architectures-from-scratch
  
2. Install dependencies (recommended: create a virtual environment)
    ```bash
    pip install torch torchvision tqdm

3. Start training a architechture.
    Run an example (e.g., ResNet on CIFAR-10 or ImageNet-style dataset)
    ```bash
    python src/training_pipeline.py

<citations>
    <paper>
        <title>EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks</title>
        <authors>Mingxing Tan, Quoc Le</authors>
        <conference>ICML</conference>
        <year>2019</year>
    </paper>

    <paper>
        <title>Deep Residual Learning for Image Recognition</title>
        <authors>Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun</authors>
        <conference>CVPR</conference>
        <year>2016</year>
    </paper>
</citations>
