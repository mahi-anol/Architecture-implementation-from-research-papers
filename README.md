<?xml version="1.0" encoding="UTF-8"?>
<repository>
    <title>Architecture Implementation from Research Papers</title>

    <description>
        This repository contains from-scratch implementations of popular deep learning
        architectures built while studying influential research papers in Machine Learning
        and Deep Learning. The project focuses on learning-by-implementation and follows
        modular, clean, and extensible software design principles.
    </description>

    <structure>
        <overview>
            Each research paper is implemented in its own dedicated folder
            to ensure isolation, clarity, and ease of experimentation.
        </overview>

        <exampleFolders>
            <folder>efficientNet</folder>
            <folder>resnet</folder>
            <folder>unet</folder>
        </exampleFolders>
    </structure>

    <paperFolderStructure>
        <file>__init__.py</file>
        <file>config.py</file>
        <file>data_pipeline.py</file>
        <file>exception.py</file>
        <file>logger.py</file>
        <file>main.py</file>
        <file>model.py</file>
        <file>training_pipeline.py</file>
        <file>utils.py</file>
    </paperFolderStructure>

    <fileDescriptions>
        <file name="model.py">
            Contains the neural network architecture implementation from scratch.
        </file>

        <file name="training_pipeline.py">
            Manages training loops, validation, loss computation, and model saving.
        </file>

        <file name="data_pipeline.py">
            Handles dataset loading, preprocessing, transformations, and dataloaders.
        </file>

        <file name="config.py">
            Centralized configuration for model hyperparameters and experiment settings.
        </file>

        <file name="main.py">
            Entry point to start training or experiments.
        </file>

        <file name="utils.py">
            Helper functions used across the project such as metrics and checkpointing.
        </file>

        <file name="logger.py">
            Logging utilities for tracking experiments and debugging.
        </file>

        <file name="exception.py">
            Custom exception handling to improve error readability.
        </file>
    </fileDescriptions>

    <designPhilosophy>
        <principle>From-scratch implementations</principle>
        <principle>Readable and modular code</principle>
        <principle>Paper-faithful architectures</principle>
        <principle>Scalable and extensible pipelines</principle>
        <principle>Clear separation of concerns</principle>
    </designPhilosophy>

    <architectures>
        <architecture>EfficientNet</architecture>
        <architecture>ResNet</architecture>
        <architecture>UNet</architecture>
        <architecture>UNet++</architecture>
    </architectures>

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

    <futureWork>
        <item>Benchmarking scripts</item>
        <item>Mixed precision training</item>
        <item>Experiment tracking</item>
        <item>Transformer-based architectures</item>
        <item>Pretrained weights</item>
    </futureWork>
</repository>
