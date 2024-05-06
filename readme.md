# An Efficient Prototype-Based Clustering Approach for Edge Pruning in Graph Neural Networks to Battle Over-Smoothing

This repository contains the Pytorch implementation of our method for IJCAI 2024 paper: "An Efficient Prototype-Based Clustering Approach for Edge Pruning in Graph Neural Networks to Battle Over-Smoothing".

## Abstract

![]([YYHemich/ClusterDrop (github.com)](https://github.com/YYHemich/ClusterDrop\model-structure.png)

Topology augmentation, or edge modification, is a popular strategy to address the issue of over-smoothing in graph neural networks (GNNs). To prevent potential distortion of the original structure and dilution of node representations, an essential principle is to enhance the separability between embeddings of nodes from different classes while preserving smoothness among nodes of the same class. However, differentiating between inter-class and intra-class edges becomes arduous when class labels are unavailable or the graph is partially labeled. To address these limitations, we introduce ClusterDrop, which uses learnable prototypes for efficient clustering. By making the prototypes learnable, it incorporates supervised signals, leading to enhanced accuracy and adaptability across different graphs. Experiments on six datasets with varying graph structures demonstrate its effectiveness in alleviating over-smoothing and enhancing GNN performance.

## Requirements

Our codes are implemented with Python 3.8.17. To set up the environment, 

```
pip install requirements.txt
```

## Usage

To train a model with ClusterDrop, use

```
python ClusterDrop_main.py [--verbose]
```

Specifying `--verbose` to print the training log info.

To search hyperparameters, use

```
python optuna_ClusterDrop.py
```

