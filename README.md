# Deep Learning Architectures: From-Scratch Implementations from Research Papers

In my free time, I enjoy implementing well-known deep learning architectures entirely from scratch as a way to challenge and deepen my understanding of how they really work. This project is driven purely by curiosity and a passion for learning, rather than benchmarks or production goals.

Rebuilding these architectures layer by layer, wiring the training pipelines, and finally seeing the loss decrease is deeply rewarding. That moment when everything clicks, when the model trains correctly, gradients flow as expected, and theory turns into working code, is what motivates this repository.

Each implementation reflects my learning journey while studying the original research papers, aiming to translate ideas from theory into clean, readable, and modular code.

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

---

## 🛠️ Future Plans

- [ ] Adding more architectures.

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
    ```

## 📚 References / Citations

- **EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks**  
  Mingxing Tan, Quoc Le, ICML 2019  
  [Paper Link](https://arxiv.org/abs/1905.11946)

- **Deep Residual Learning for Image Recognition**  
  Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun, CVPR 2016  
  [Paper Link](https://arxiv.org/abs/1512.03385)

- **U-Net: Convolutional Networks for Biomedical Image Segmentation**  
  Olaf Ronneberger, Philipp Fischer, Thomas Brox, MICCAI 2015  
  [Paper Link](https://arxiv.org/abs/1505.04597)

