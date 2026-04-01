# A Comparative Analysis of CNN vs Vision Transformers

## * Authors

**Chris Essomba, Christopher BANFOU, Ivan BIBA** — 
University of Paris Cité



## * Project Overview

This project presents a comprehensive comparative analysis between **Convolutional Neural Networks (CNN)** and **Vision Transformers (ViT)** on image classification tasks. The study evaluates both architectures across different visual characteristics: texture-based features and long-range dependencies, using a curated dataset derived from Tiny-ImageNet-200.

The goal is to understand the strengths and weaknesses of each architecture in different scenarios and provide insights into their performance characteristics, computational efficiency, and learning patterns.

---

## * Key Research Questions

- How do CNNs and Vision Transformers compare in texture classification tasks?
- Which architecture better captures long-range dependencies in images?
- What are the computational trade-offs between these two approaches?
- How do the architectures perform on a limited dataset (3500 training samples)?

---

## 📊 Dataset Description

### Dataset Composition
- **Source**: Tiny-ImageNet-200 (10 classes selected)
- **Total Images**: 4,750 images
  - Training: 3,500 images
  - Validation: 750 images
  - Testing: 750 images
- **Image Size**: 64×64×3 (RGB color channels)
- **Image Quality**: Various conditions including partial occlusions, degraded images, and diverse lighting

### Class Organization
The dataset is split into two categories for targeted analysis:

#### 1. **Texture-Based Classes** (5 classes)
Classes selected for texture recognition capabilities, testing how well models capture fine-grained surface patterns and textures.

#### 2. **Long-Range Dependency Classes** (5 classes)
Classes requiring the model to understand spatial relationships and dependencies across distant image regions.

Both categories are further organized in the `generated_datasets/` directory into:
```
generated_datasets/
├── texture_dataset/
│   ├── train/
│   ├── test/
│   └── validation/
└── global_dataset/
    ├── train/
    ├── test/
    └── validation/
```

---

## 🏗️ Model Architectures

### 1. Convolutional Neural Network (CNN)

**Purpose**: Extract spatial features through hierarchical convolution operations.

**Architecture Overview**:
- **Input**: 64×64×3 RGB images
- **Feature Extractor**: 
  - Multiple convolutional blocks with kernels of varying sizes
  - ReLU activations for non-linearity
  - Max pooling for spatial dimension reduction
  - Batch normalization for stable training
- **Classifier**: 
  - Fully connected layers
  - Softmax output for 10-class classification
- **Parameters**: ~295K parameters
- **Key Characteristics**:
  - Strong inductive bias for spatial locality
  - Efficient parameter sharing through convolution
  - Good for capturing local texture patterns
  - Limited receptive field for long-range dependencies

**Performance**: CNN shows strong performance on texture-based classification tasks due to its hierarchical feature learning and local receptive fields.

### 2. Vision Transformer (ViT)

**Purpose**: Apply transformer self-attention mechanisms to image classification through patch-based tokenization.

**Architecture Overview**:

#### Stage 1: Patch Embedding & Projection
- **Input Image**: 64×64×3
- **Patch Size**: 8×8 pixels → 64 patches (8×8 grid)
- **Embedding Dimension**: 192
- **Projection Method**: Hybrid convolution + neighbor feature aggregation
  - Captures spatial locality while creating patch-level embeddings
  - Dynamic feature aggregation from neighboring patches
  - Reduces information loss from direct flattening

#### Stage 2: Sequence Construction
- **CLS Token**: A learnable 192-dim vector prepended to the patch sequence
  - Acts as a global information receptor
  - Used for final classification decision
- **Positional Embeddings**: Shape [1, 65, 192]
  - Preserves spatial order of patches
  - Allows transformer to distinguish patch positions
- **Output Sequence**: 65 tokens (1 CLS + 64 patches)

#### Stage 3: Transformer Encoder
- **Number of Layers**: 4 identical TransformerEncoderLayer blocks
- **Multi-Head Attention**:
  - Number of heads: 8
  - Dimension per head: 24 (192÷8)
  - Allows multiple representation subspaces
- **Feed-Forward Network**:
  - Expansion: 192 → 768 → 192
  - Activation: GELU (Gaussian Error Linear Unit)
  - Per-token transformation
- **Normalization**: Pre-LayerNorm applied before each sub-block
- **Regularization**: 25% Dropout
  - Aggressive regularization suited for small datasets
  - Prevents overfitting on limited training data

#### Stage 4: Classification Head
- **Feature Extraction**: Only the refined CLS token
- **Projection**: Convolutional MLP
  - Projects from 192 dimensions to 10 class logits
  - Produces final classification predictions

**Key Characteristics**:
- Global receptive field from the start (can see all patches at once)
- Strong for capturing long-range dependencies
- Requires more data for optimal learning
- Better representational capacity for complex patterns
- Fewer spatial inductive biases compared to CNN

**Model Checkpoints**: Pre-trained ViT models available in `modeles/`:
- `ViT_modele_all.ckpt` - Trained on combined dataset
- `ViT_modele_texture.ckpt` - Specialized for texture classification
- `ViT_modele_long_range.ckpt` - Specialized for long-range dependencies

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- CUDA-capable GPU (recommended for faster training)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd Analyse_compartive_CNN_Transformers
```

### Step 2: Create a Virtual Environment (Optional but Recommended)
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n cnn-vit-analysis python=3.10
conda activate cnn-vit-analysis
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Required Libraries
The project uses:
- **PyTorch**: Deep learning framework (`torch`, `torchvision`)
- **PyTorch Lightning**: Training framework (`pytorch_lightning`, `lightning`)
- **Data & Processing**: `pandas`, `numpy`, `scikit-learn`
- **Metrics**: `torchmetrics`
- **Visualization**: `matplotlib`, `pillow`
- **Utilities**: `tqdm`, `jinja2`, `pdb`

---

## 📈 Usage & Running the Project

### Running the CNN Model
```bash
jupyter notebook notebook/TP_CNN.ipynb
```

The CNN notebook includes:
1. Data loading from the `generated_datasets/` directory
2. Model architecture definition
3. Training loop with validation monitoring
4. Metrics tracking and visualization
5. Results saved to `Train_test_results/CNN/`

### Running the Vision Transformer Model
```bash
jupyter notebook notebook/TP_VIT.ipynb
```

The ViT notebook includes:
1. Patch embedding and dataset preparation
2. Vision Transformer architecture implementation
3. Training with PyTorch Lightning
4. Early stopping and learning rate scheduling
5. Comprehensive metrics and confusion matrices
6. Model checkpointing in `modeles/`

### Expected Outputs
After running the notebooks, you'll generate:
- Training/validation loss and accuracy curves
- Confusion matrices for test sets
- Classification metrics (Precision, Recall, F1-Score)
- Model checkpoints for future inference

---

## 📊 Results & Findings

### Directory Structure
```
Train_test_results/
├── CNN/
│   ├── cnn_texture_results.json      # Texture dataset results
│   └── cnn_long_range_results.json   # Long-range dataset results
└── ViT/
    ├── confusion_matrices_all.json
    ├── confusion_matrices_texture.json
    ├── confusion_matrices_long_range.json
    ├── test_confusion_matrix_all.json
    ├── test_confusion_matrix_texture.json
    └── test_confusion_matrix_long_range.json
```

### Key Metrics
Each result file contains:
- **Model Architecture**: CNN or ViT
- **Dataset**: texture, long_range, or all
- **Parameters**: Total number of trainable parameters
- **Training History**: Per-epoch metrics (loss, accuracy, F1-score)
- **Learning Rate**: Optimizer settings
- **Confusion Matrices**: Detailed per-class performance

### Performance Visualization
Visual comparisons are available in `metriques/`:
- Architecture diagrams for both CNN and ViT
- Learning curves comparing training/validation metrics
- Confusion matrices for different dataset categories
- Detailed analysis of classification errors

---

## 📁 Project Structure

```
Analyse_compartive_CNN_Transformers/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── notebook/                           # Jupyter notebooks
│   ├── TP_CNN.ipynb                   # CNN training & evaluation
│   └── TP_VIT.ipynb                   # ViT training & evaluation
├── generated_datasets/                 # Dataset splits
│   ├── texture_dataset/               # Texture-focused images
│   │   ├── train/
│   │   └── test/
│   └── global_dataset/                # All images combined
│       ├── train/
│       └── test/
├── modeles/                            # Trained model checkpoints
│   ├── ViT_modele_all.ckpt
│   ├── ViT_modele_texture.ckpt
│   └── ViT_modele_long_range.ckpt
├── Train_test_results/                # Training results & metrics
│   ├── CNN/
│   │   ├── cnn_texture_results.json
│   │   └── cnn_long_range_results.json
│   └── ViT/
│       ├── confusion_matrices_*.json
│       └── test_confusion_matrix_*.json
├── metriques/                          # Visualizations & diagrams
│   ├── archi_CNN.png
│   ├── archi_VIT.png
│   ├── learning_curves_*.png
│   ├── confusion_matrix_*.png
│   └── architecture_vit_generee/
└── notes/                              # Documentation
    ├── ViT.md                         # ViT architecture details
    └── explication_cnn_notebook.txt   # CNN notebook explanation
```

---

## 🔍 Detailed Architecture Notes

### Vision Transformer Details
See [notes/ViT.md](notes/ViT.md) for comprehensive architecture documentation including:
- Patch embedding strategies
- Attention mechanism details
- Positional encoding
- Multi-head attention configuration
- Feed-forward network design
- Pre-training consideration

### CNN Architecture Details
See [notes/explication_cnn_notebook.txt](notes/explication_cnn_notebook.txt) for:
- Convolutional block structure
- Feature extraction pipeline
- Classification head design
- Hyperparameter selection

---

## 💡 Key Insights & Recommendations

### When to Use CNN:
- ✅ Small datasets with strong local patterns
- ✅ Texture and fine-grained feature classification
- ✅ Computationally constrained environments
- ✅ Cases where spatial locality is important

### When to Use Vision Transformer:
- ✅ Capturing long-range spatial dependencies
- ✅ Sufficient training data available
- ✅ Complex scene understanding required
- ✅ When global context is crucial for classification

### Trade-offs:
| Aspect | CNN | ViT |
|--------|-----|-----|
| **Inductive Bias** | Strong (locality) | Weak (data-driven) |
| **Receptive Field** | Grows with depth | Global from start |
| **Data Requirements** | Low to medium | Medium to high |
| **Computational Cost** | Low | High |
| **Texture Performance** | Strong | Moderate |
| **Long-range Dependencies** | Moderate | Strong |
| **Parameter Efficiency** | High | Medium |

---

## 📦 Model Checkpoints

Pre-trained models are available in the `modeles/` directory:

### Loading a Model
```python
import torch
from lightning.pytorch import Trainer

# Load ViT checkpoint
checkpoint = torch.load('modeles/ViT_modele_texture.ckpt')

# Use with Lightning
trainer = Trainer()
# Load for inference
model = trainer.validate(checkpoint)
```

---

## 🛠️ Development Notes

- **Framework**: PyTorch & PyTorch Lightning
- **Training Approach**: Supervised learning with cross-entropy loss
- **Optimization**: Adam optimizer with learning rate scheduling
- **Regularization**: Dropout, layer normalization, early stopping
- **Validation**: Cross-validation with separate test sets
- **Tracking**: Comprehensive metric logging and visualization

---

## 📚 References & Resources

- Vision Transformers (ViT): [Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929)
- Tiny-ImageNet-200: [Standard benchmark dataset](https://www.image-net.org/download.php)
- PyTorch Lightning: [Official Documentation](https://lightning.ai/)
- Attention Mechanisms: [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

---

## 🤝 Contributing

For questions, suggestions, or improvements:
1. Review the existing notebooks and documentation
2. Check the `notes/` directory for architecture details
3. Compare results in `Train_test_results/`
4. Modify notebooks for new experiments

---

## 📝 License

This project is part of a Deep Learning study initiative. Please respect the original dataset licenses (Tiny-ImageNet-200).

---

## 📧 Contact & Support

For questions about this project or reproducibility issues, please refer to the detailed notes in the `notes/` directory or review the Jupyter notebooks for implementation details.

---

**Last Updated**: April 2026
**Status**: Analysis Complete with Results Generated