# Spotscape

<p align="center">   
    <a href="https://pytorch.org/" alt="PyTorch">
      <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white" /></a>
    <a href="https://icml.cc/Conferences/2025" alt="Conference">
        <img src="https://img.shields.io/badge/ICML'25-brightgreen" /></a>
    <img src="https://img.shields.io/pypi/l/torch-rechub">
</p>

The official source code for [Global Context-aware Representation Learning for Spatially Resolved Transcriptomics](https://openreview.net/forum?id=jeJGH6UDOL), accepted at ICML 2025.

## Overview

Spatially Resolved Transcriptomics (SRT) is a cutting-edge technique that captures the spatial context of cells within tissues, enabling the study of complex biological networks. Recent graph-based methods leverage both gene expression and spatial information to identify relevant spatial domains. However, these approaches fall short in obtaining meaningful spot representations, especially for spots near spatial domain boundaries, as they heavily emphasize adjacent spots that have minimal feature differences from an anchor node. To address this, we propose **Spotscape**, a novel framework that introduces the Similarity Telescope module to capture global relationships between multiple spots. Additionally, we propose a similarity scaling strategy to regulate the distances between intra- and inter-slice spots, facilitating effective multi-slice integration. Extensive experiments demonstrate the superiority of **Spotscape** in various downstream tasks, including single-slice and multi-slice scenarios.


<figure>
  <p style="text-align: center;">
  <img src="imgs/Architecture.png" width="500px" />
  <figcaption>Figure 1. Overall framework of Spotscape, which is trained with the spatial nearest neighbor (SNN) graph, using (a) similary telescope and reconstruction loss, while additionally utilizing (b) PCL and similarity scale matching loss in multi-slice spatially resolved transcriptomics</figcaption>
  </p>
</figure>


## Environments

* ``python`` version: 3.9.7
* ``pytorch`` version: 2.1.1
* ``pyg`` version: 2.4.0
* ``pyg_lib`` version: 0.3.1
* ``faiss-cpu`` version: 1.7.4

```bash
# If your conda is not up-to-date (optional)
conda update -n base conda --yes
conda update --all --yes

# For environment setting
conda create --name spotscape python=3.9.7 --yes
conda activate spotscape
sh environment_setup.sh
```

## Datasets

Please find the datasets in this [link](https://drive.google.com/drive/folders/18T6pYS4HW08F03xnK_l7gGCRwXtkMGSO?usp=sharing).

| **Dataset** | **Species** | **Tissue**                               | **Technology** | **Resolution** | **Cells/Spots**          | **Genes**                | **\# of Spatial Domains** | **Reference**             |
|------------------|------------------|-----------------------------------------------|---------------------|---------------------|-------------------------------|-------------------------------|--------------------------------|--------------------------------|
| 10x_dlpfc            | Human            | Brain (dorsolateral prefrontal cortex; DLPFC) | 10x Visium          | 50 μm           | 3460 ~ 4789   | 33538                         | 5 ~ 7          | [link](https://www.nature.com/articles/s41593-020-00787-0)                  |
| mtg              | Human            | Brain (middle temporal gyrus; MTG)            | 10x Visium          | 50 μm           | 3445 ~ 4832   | 36601                         | 6 ~ 7          | [link](https://actaneurocomms.biomedcentral.com/articles/10.1186/s40478-022-01494-6)                 |
| mouse_embryo     | Mouse            | Whole embryo                                  | Stereo-seq          | 0.2 μm          | 30756 ~ 55295 | 25485 ~ 27330 | 18 ~ 19        | [link](https://www.cell.com/cell/fulltext/S0092-8674(22)00399-3?dgcid=raven_jbs_etoc_email) |
| nsclc            | Human            | Non-small cell lung cancer (NSCLC)            | NanoString CosMX    | Subcellular         |    11756                        | 960                         | 4                              | [link](https://link.springer.com/article/10.1186/s13059-024-03241-7)                  |




## How to run the code?

Please find the example shell files as below.

```bash
# single-slice
sh run_single.sh

# multi-slice
sh run_multi_integration.sh

# large multi-slice (e.g., integration of 10 slices) for reducing the computational complexity
sh run_multi_integration_large.sh
```

For the detailed descriptions of its arguments, please refer to the ``./utils/argument.py`` file.

## Cite

- If you use **Spotscape** in your research, please cite the following paper:
  - Yunhak Oh*, Junseok Lee*, Yeongmin Kim, Sangwoo Seo, Namkyeong Lee, and Chanyoung Park. "Global Context-aware Representation Learning for Spatially Resolved Transcriptomics." ICML 2025.

```bibtex
@inproceedings{
    oh2025global,
    title={Global Context-aware Representation Learning for Spatially Resolved Transcriptomics},
    author={Yunhak Oh and Junseok Lee and Yeongmin Kim and Sangwoo Seo and Namkyeong Lee and Chanyoung Park},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=jeJGH6UDOL}
}
```