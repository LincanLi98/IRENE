# IRENE: Information Bottleneck Guided Graph Learning for Interpretable EEG Seizure Detection

## Project Overview

This repository contains the official implementation of **IRENE**, a novel framework for interpretable seizure detection from EEG signals via information-bottleneck guided dynamic graph learning.

IRENE jointly learns:

* A denoised and compact dynamic brain graph structure.
* Spatio-temporal EEG representations guided by clinical relevance.

The model is designed to tackle the core challenges of EEG-based seizure detection:

* High noise in EEG signals
* Extreme inter-patient variability
* Label scarcity

## 🧪 Key Features

* **IB-Guided Graph Constructor**: Learns sparse self-expressive dynamic graphs.
* **Graph Structure Aware Transformer Encoder**: Integrates learned structure with full attention.
* **Masked Graph AutoEncoder Pretraining**: Enables robust unsupervised representation learning.
* **Support for TUH and CHB-MIT datasets**

## 📁 Repository Structure

```
├── model/
│   ├── GSA_Encoder.py         # Structure-aware encoder (Transformer backbone)
│   ├── GraphMAE.py            # Masked graph autoencoder pretrainer
│   ├── IRENE.py               # Full IRENE classification model
│   ├── graph_constructor.py   # IB-guided dynamic graph construction module
│   ├── loss.py                # IRENE total loss (IB + Smooth + Reconstruction)
│   ├── lstm.py                # LSTM and CNN-LSTM baseline models
│   ├── DCRNN.py               # DCRNN baseline model
│   ├── NeuroGNN.py            # NeuroGNN baseline model
│   ├── GraphS4mer.py          # GraphS4mer baseline model
│   └── cnnlstm.py             # ResNet-LSTM baseline model
├── data/
│   ├── dataloader_*.py        # Dataset-specific loaders for TUSZ, CHB-MIT
│   ├── preprocess_*.py        # Preprocessing scripts for FFT, segmenting, graph construction
│   └── electrode_graph/       # Precomputed adjacency matrices
├── train/finetune.py          # Fine-tuning script after pretraining
├── main.py                    # Entry point for training/evaluation
├── args.py                    # Full argument parser
├── utils.py                   # Logging, checkpoint, metrics utilities
├── processed_data/            # folder to place the preprocessed EEG data from TUSZ, which will then sent into models
├── TUSZ_v1.5.2/               # folder to place your raw TUSZ v1.5.2 dataset
```

## 🚀 Getting Started

### Prerequisites

* Python 3.8+
* PyTorch 1.13+

Install dependencies:

```bash
pip install -r requirements.txt
```

### Datasets

This repo supports:

* [TUH Seizure Corpus (TUSZ)](https://www.isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml)

Preprocessed EEG (FFT or raw) should be placed in:

```
processed_data/
```

**Data preprocessing scripts can be found under `data/`:**

* Segment EEG into time windows
* Perform frequency transformation (e.g., FFT)
* Normalize & generate electrode graphs (adjacency matrices)

### Training IRENE

```bash
python main.py --model_name IRENE --dataset TUSZ --task detection \
  --input_dim 100 --embed_dim 64 \
  --lambda1 0.5 --lambda2 0.5 --lambda3 0.5 --lambda4 0.5
```

### Running Baselines

You can easily run baseline methods by changing `--model_name`, for example:

```bash
# LSTM baseline
python main.py --model_name lstm --dataset TUSZ --task detection

# CNN-LSTM baseline
python main.py --model_name cnnlstm --dataset TUSZ --task detection

# DCRNN
python main.py --model_name dcrnn --dataset TUSZ --task detection

# EvolveGCN
python main.py --model_name evolvegcn --dataset TUSZ --task detection

# BIOT
python main.py --model_name BIOT --dataset TUSZ --task detection
```

Each model supports its own configuration through arguments defined in `args.py`.

### Fine-tuning from Pretrained

```bash
python train/finetune.py --load_model_path path/to/pretrained.pth --fine_tune
```

## 📊 Results

IRENE achieves state-of-the-art performance on TUSZ benchmark, while also providing interpretable & dynamic brain graphs and channel dependencies.

## 🧩 Citation

If you find this work useful, please kindly cite our paper:

```bibtex
@article{2025irene,
  title={Information Bottleneck Guided Graph Learning for Interpretable EEG Seizure Detection},
  author={AAA, BBB, CCC, DDD, and EEE},
  journal={Openreview},
  year={2025}
}
```


## 📬 Contact

<!-- For questions or collaborations, please contact [Lincan Li](mailto:ll24bb@fsu.edu).-->
For questions or collaborations, please contact us.



