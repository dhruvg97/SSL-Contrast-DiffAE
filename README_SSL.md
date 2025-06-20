# Self-Supervised Contrastive Diffusion Autoencoder for Medical Image Phenotype Discovery

This repository extends the original Contrast-DiffAE with dedicated self-supervised learning capabilities for medical image analysis, specifically designed for phenotype discovery in chest X-rays and CT scans.

## 🎯 Project Goals

### Phase 1: Validation with VinDR Chest X-ray Dataset
- Validate self-supervised phenotype discovery using known pathologies
- 14 pathology classes for cluster validation
- Higher resolution images (352x352 or 512x512)
- t-SNE visualization and clustering analysis

### Phase 2: Discovery in Sarcoidosis CT (Future)
- Apply validated methods to proprietary CT dataset
- Discover novel sarcoidosis phenotypes
- Fully unsupervised phenotype extraction

## 🏗️ Architecture Overview

### Key Components
1. **SelfSupervisedLitModel**: Dedicated training class for self-supervised learning
2. **VinDRChestXrayDataset**: Medical image dataset with augmentation support
3. **InfoNCE Contrastive Loss**: Self-supervised learning using augmented pairs
4. **Analysis Tools**: t-SNE, clustering, and prototype extraction

### Self-Supervised Strategy
- **Positive pairs**: Original and augmented versions of the same image
- **Negative pairs**: Different patients/images
- **No labels used during training**: Only for validation analysis
- **Contrastive learning**: InfoNCE loss encourages similar representations for augmented pairs

## 📁 New Files

### Core Implementation
- `experiment_ssl.py`: Self-supervised training class and pipeline
- `train_ssl.py`: Training script for self-supervised learning
- `analyze_ssl_results.py`: Comprehensive analysis tools

### Configuration
- `templates.py`: Added VinDR SSL configurations
- `config.py`: Extended with SSL parameters
- `dataset.py`: Added VinDRChestXrayDataset class

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Install additional dependencies for SSL
pip install umap-learn seaborn scikit-learn
```

### 2. Prepare VinDR Dataset
```bash
# Download VinDR-CXR dataset from: https://vindr.ai/datasets/cxr
# Structure should be:
/path/to/vindr/
├── images/
│   ├── image1.png
│   └── ...
├── train.csv
├── val.csv
└── test.csv
```

### 3. Configure Training
```python
# Edit train_ssl.py
conf.vindr_data_path = '/path/to/your/vindr/dataset'  # UPDATE THIS
conf.img_size = 352  # or 512 for higher resolution
conf.batch_size = 8  # Adjust based on GPU memory
```

### 4. Start Training
```bash
python train_ssl.py
```

### 5. Monitor Training
```bash
tensorboard --logdir=checkpoints
```

## 📊 Analysis and Visualization

### During Training
- **Automatic t-SNE plots**: Generated every 10 epochs
- **Clustering metrics**: ARI and NMI scores logged to TensorBoard
- **Contrastive loss tracking**: Monitor convergence

### Post-Training Analysis
```bash
python analyze_ssl_results.py \
    --model_path checkpoints/VinDR_SSL_352/checkpoints/ssl-epoch-XX.ckpt \
    --config VinDR_SSL_352 \
    --split test \
    --save_dir results_analysis
```

### Analysis Outputs
- `clustering_metrics.csv`: Quantitative clustering performance
- `dimensionality_reduction_comparison.png`: t-SNE, UMAP, PCA, MDS visualizations
- `prototypes.csv`: Most representative images for each pathology
- `tsne_plots/`: Time series of t-SNE evolution during training

## ⚙️ Configuration Options

### Image Resolution
```python
# For 352x352 (recommended for initial experiments)
conf = VinDR_SSL_352()

# For 512x512 (requires more GPU memory)
conf = VinDR_SSL_512()
```

### Self-Supervised Parameters
```python
conf.augmentation_strength = 0.8    # 0.0 to 1.0
conf.alpha = 1.0                    # Contrastive loss weight
conf.load_in = 10                   # Epochs before starting contrastive learning
conf.compute_tsne = True            # Generate t-SNE visualizations
conf.tsne_perplexity = 30          # t-SNE perplexity parameter
```

### Medical Image Augmentations
- Random affine transformations (rotation, translation, scaling)
- Horizontal flips
- Color jitter (brightness, contrast)
- Gaussian noise injection
- Controlled by `augmentation_strength` parameter

## 📈 Expected Results

### Training Progression
1. **Phase 1 (0-10 epochs)**: Pure diffusion autoencoder training
2. **Phase 2 (10+ epochs)**: Addition of contrastive learning
3. **Validation**: Clustering metrics improve over time
4. **Visualization**: t-SNE plots show increasing pathology separation

### Success Metrics
- **ARI (Adjusted Rand Index)**: >0.3 indicates good clustering
- **NMI (Normalized Mutual Information)**: >0.4 shows semantic alignment
- **Visual inspection**: Clear pathology clusters in t-SNE plots
- **Prototype coherence**: Representative images make clinical sense

## 🔬 Advanced Analysis

### Clustering Methods
- K-means clustering (15 clusters for 14 pathologies + no finding)
- DBSCAN for density-based clustering
- Silhouette score for cluster quality assessment

### Dimensionality Reduction
- **t-SNE**: Non-linear, good for visualization
- **UMAP**: Preserves global structure
- **PCA**: Linear baseline
- **MDS**: Distance preservation

### Prototype Extraction
- Find images closest to cluster centroids
- Export prototype image IDs for clinical review
- Quantify prototype representativeness

## 🛠️ Troubleshooting

### Common Issues

**GPU Memory Issues (512x512)**
```python
conf.batch_size = 2  # Reduce batch size
conf.net_ch = 32     # Use smaller model
```

**Poor Clustering Performance**
- Increase `augmentation_strength`
- Adjust `alpha` (contrastive loss weight)
- Try different `load_in` values
- Ensure dataset balance

**t-SNE Visualization Problems**
- Reduce `max_points` in visualization
- Adjust `tsne_perplexity` (10-50 range)
- Check for NaN values in embeddings

## 📋 Next Steps for Phase 2 (Sarcoidosis CT)

1. **Adapt dataset loader** for CT volumes or 2D slices
2. **Modify augmentations** for CT-specific transforms
3. **Implement 3D analysis** if using volumetric data
4. **Clinical validation** of discovered phenotypes
5. **Outcome correlation** with clinical metadata

## 🤝 Usage Examples

### Basic Training
```python
from templates import VinDR_SSL_352
from experiment_ssl import train_self_supervised

conf = VinDR_SSL_352()
conf.vindr_data_path = '/path/to/vindr'
model, trainer = train_self_supervised(conf, gpus=[0])
```

### Custom Configuration
```python
conf = VinDR_SSL_352()
conf.augmentation_strength = 0.9  # Stronger augmentations
conf.alpha = 2.0                  # Higher contrastive weight
conf.batch_size = 4               # Smaller batches
```

### Analysis Only
```python
from analyze_ssl_results import SSLResultsAnalyzer

analyzer = SSLResultsAnalyzer('path/to/checkpoint.ckpt', 'VinDR_SSL_352')
results = analyzer.analyze_all(split='test', save_dir='my_analysis')
```

## 📊 Performance Benchmarks

### VinDR-CXR Expected Performance
- **Training time**: ~2-3 days on single GPU (352x352)
- **Memory usage**: ~8GB GPU memory (batch_size=8, 352x352)
- **Clustering ARI**: 0.25-0.40 (depending on hyperparameters)
- **t-SNE separation**: Visible pathology clusters after 50+ epochs

This implementation provides a solid foundation for self-supervised medical image analysis with clear pathways for validation and discovery applications. 