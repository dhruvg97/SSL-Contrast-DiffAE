"""
Utility script for analyzing self-supervised learning results
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import umap
from pathlib import Path
import argparse

from templates import *
from experiment_ssl import SelfSupervisedLitModel
from dataset import VinDRChestXrayDataset


class SSLResultsAnalyzer:
    """Analyzer for self-supervised learning results"""
    
    def __init__(self, model_path, config_name='VinDR_SSL_352'):
        """
        Args:
            model_path: Path to trained model checkpoint
            config_name: Name of the configuration used for training
        """
        self.model_path = model_path
        self.config_name = config_name
        
        # Load configuration
        if config_name == 'VinDR_SSL_352':
            self.conf = VinDR_SSL_352()
        elif config_name == 'VinDR_SSL_512':
            self.conf = VinDR_SSL_512()
        else:
            raise ValueError(f"Unknown config: {config_name}")
        
        # Load model
        self.model = SelfSupervisedLitModel.load_from_checkpoint(
            model_path, conf=self.conf
        )
        self.model.eval()
        
    def extract_embeddings(self, split='test', max_samples=None):
        """Extract embeddings from the specified split"""
        
        # Update dataset configuration for analysis
        self.conf.vindr_split = split
        dataset = self.conf.make_dataset()
        dataset.return_labels = True
        dataset.self_supervised = False
        
        if max_samples:
            dataset = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))
        
        loader = self.conf.make_loader(
            dataset, shuffle=False, batch_size=32, drop_last=False
        )
        
        embeddings_list = []
        pathology_labels_list = []
        dominant_pathology_list = []
        image_ids_list = []
        
        with torch.no_grad():
            for batch in loader:
                imgs = batch['img']
                
                # Extract embeddings
                embeddings = self.model.model.encoder(imgs)
                embeddings_list.append(embeddings.cpu())
                
                # Store labels and metadata
                pathology_labels_list.append(batch['pathology_labels'])
                dominant_pathology_list.append(batch['dominant_pathology'])
                image_ids_list.append(batch['image_id'])
        
        # Concatenate results
        embeddings = torch.cat(embeddings_list, dim=0).numpy()
        pathology_labels = torch.cat(pathology_labels_list, dim=0).numpy()
        dominant_pathology = torch.cat(dominant_pathology_list, dim=0).numpy()
        image_ids = sum(image_ids_list, [])  # Flatten list of lists
        
        return {
            'embeddings': embeddings,
            'pathology_labels': pathology_labels,
            'dominant_pathology': dominant_pathology,
            'image_ids': image_ids,
            'pathology_names': self.conf.vindr_pathologies + ('No Finding',)
        }
    
    def clustering_analysis(self, embeddings_data, save_dir='ssl_analysis'):
        """Perform comprehensive clustering analysis"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        embeddings = embeddings_data['embeddings']
        dominant_pathology = embeddings_data['dominant_pathology']
        pathology_names = embeddings_data['pathology_names']
        
        results = {}
        
        # Try different clustering algorithms
        clustering_methods = {
            'kmeans': KMeans(n_clusters=len(pathology_names), random_state=42),
            'dbscan': DBSCAN(eps=0.5, min_samples=5),
        }
        
        for method_name, clusterer in clustering_methods.items():
            print(f"Running {method_name} clustering...")
            
            cluster_labels = clusterer.fit_predict(embeddings)
            
            # Compute metrics
            if len(np.unique(cluster_labels)) > 1:
                ari = adjusted_rand_score(dominant_pathology, cluster_labels)
                nmi = normalized_mutual_info_score(dominant_pathology, cluster_labels)
                silhouette = silhouette_score(embeddings, cluster_labels)
            else:
                ari = nmi = silhouette = 0.0
            
            results[method_name] = {
                'cluster_labels': cluster_labels,
                'ari': ari,
                'nmi': nmi,
                'silhouette': silhouette,
                'n_clusters': len(np.unique(cluster_labels))
            }
            
            print(f"  ARI: {ari:.4f}")
            print(f"  NMI: {nmi:.4f}")
            print(f"  Silhouette: {silhouette:.4f}")
            print(f"  N clusters: {len(np.unique(cluster_labels))}")
        
        # Save results
        results_df = pd.DataFrame({
            method: {
                'ARI': results[method]['ari'],
                'NMI': results[method]['nmi'],
                'Silhouette': results[method]['silhouette'],
                'N_Clusters': results[method]['n_clusters']
            }
            for method in results.keys()
        }).T
        
        results_df.to_csv(save_dir / 'clustering_metrics.csv')
        print(f"Clustering metrics saved to {save_dir / 'clustering_metrics.csv'}")
        
        return results
    
    def dimensionality_reduction_visualization(self, embeddings_data, save_dir='ssl_analysis'):
        """Create multiple dimensionality reduction visualizations"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        embeddings = embeddings_data['embeddings']
        dominant_pathology = embeddings_data['dominant_pathology']
        pathology_names = embeddings_data['pathology_names']
        
        # Subsample for visualization if too large
        max_points = 3000
        if len(embeddings) > max_points:
            indices = np.random.choice(len(embeddings), max_points, replace=False)
            embeddings_viz = embeddings[indices]
            pathology_viz = dominant_pathology[indices]
        else:
            embeddings_viz = embeddings
            pathology_viz = dominant_pathology
        
        # Different reduction methods
        reduction_methods = {
            'tsne': TSNE(n_components=2, perplexity=30, random_state=42),
            'umap': umap.UMAP(n_components=2, random_state=42),
            'pca': PCA(n_components=2),
            'mds': MDS(n_components=2, random_state=42)
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for idx, (method_name, reducer) in enumerate(reduction_methods.items()):
            print(f"Computing {method_name.upper()}...")
            
            # Compute embedding
            if method_name == 'umap':
                embeddings_2d = reducer.fit_transform(embeddings_viz)
            else:
                embeddings_2d = reducer.fit_transform(embeddings_viz)
            
            # Plot
            ax = axes[idx]
            colors = plt.cm.tab20(np.linspace(0, 1, len(pathology_names)))
            
            for i, (pathology, color) in enumerate(zip(pathology_names, colors)):
                mask = pathology_viz == i
                if mask.sum() > 0:
                    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                             c=[color], label=pathology, alpha=0.7, s=20)
            
            ax.set_title(f'{method_name.upper()} Visualization')
            ax.set_xlabel(f'{method_name.upper()} Component 1')
            ax.set_ylabel(f'{method_name.upper()} Component 2')
            
            if idx == 0:  # Only show legend for first plot
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'dimensionality_reduction_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Dimensionality reduction plots saved to {save_dir / 'dimensionality_reduction_comparison.png'}")
    
    def extract_prototypes(self, embeddings_data, n_prototypes=5, save_dir='ssl_analysis'):
        """Extract prototype images for each pathology"""
        
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        embeddings = embeddings_data['embeddings']
        dominant_pathology = embeddings_data['dominant_pathology']
        pathology_names = embeddings_data['pathology_names']
        image_ids = embeddings_data['image_ids']
        
        prototypes = {}
        
        for pathology_idx, pathology_name in enumerate(pathology_names):
            # Get embeddings for this pathology
            mask = dominant_pathology == pathology_idx
            if mask.sum() == 0:
                continue
                
            pathology_embeddings = embeddings[mask]
            pathology_image_ids = [image_ids[i] for i in np.where(mask)[0]]
            
            # Find cluster center
            center = pathology_embeddings.mean(axis=0)
            
            # Find closest images to center
            distances = np.linalg.norm(pathology_embeddings - center, axis=1)
            closest_indices = np.argsort(distances)[:n_prototypes]
            
            prototype_ids = [pathology_image_ids[i] for i in closest_indices]
            prototype_distances = distances[closest_indices]
            
            prototypes[pathology_name] = {
                'image_ids': prototype_ids,
                'distances': prototype_distances,
                'center': center
            }
            
            print(f"{pathology_name}: {len(pathology_embeddings)} samples, "
                  f"prototypes: {prototype_ids[:3]}...")
        
        # Save prototype information
        prototype_df = pd.DataFrame([
            {
                'pathology': pathology,
                'prototype_rank': rank + 1,
                'image_id': img_id,
                'distance_to_center': dist
            }
            for pathology, data in prototypes.items()
            for rank, (img_id, dist) in enumerate(zip(data['image_ids'], data['distances']))
        ])
        
        prototype_df.to_csv(save_dir / 'prototypes.csv', index=False)
        print(f"Prototype information saved to {save_dir / 'prototypes.csv'}")
        
        return prototypes
    
    def analyze_all(self, split='test', max_samples=None, save_dir='ssl_analysis'):
        """Run complete analysis pipeline"""
        
        print("Starting comprehensive SSL analysis...")
        print(f"Model: {self.model_path}")
        print(f"Split: {split}")
        
        # Extract embeddings
        print("\n1. Extracting embeddings...")
        embeddings_data = self.extract_embeddings(split, max_samples)
        print(f"Extracted {len(embeddings_data['embeddings'])} samples")
        
        # Clustering analysis
        print("\n2. Performing clustering analysis...")
        clustering_results = self.clustering_analysis(embeddings_data, save_dir)
        
        # Dimensionality reduction visualization
        print("\n3. Creating visualizations...")
        self.dimensionality_reduction_visualization(embeddings_data, save_dir)
        
        # Extract prototypes
        print("\n4. Extracting prototypes...")
        prototypes = self.extract_prototypes(embeddings_data, save_dir=save_dir)
        
        print(f"\nAnalysis completed! Results saved in: {save_dir}")
        print(f"Key files:")
        print(f"  - clustering_metrics.csv: Clustering performance metrics")
        print(f"  - dimensionality_reduction_comparison.png: Visualization comparison")
        print(f"  - prototypes.csv: Prototype image information")
        
        return {
            'embeddings_data': embeddings_data,
            'clustering_results': clustering_results,
            'prototypes': prototypes
        }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze SSL results')
    parser.add_argument('--model_path', required=True, help='Path to model checkpoint')
    parser.add_argument('--config', default='VinDR_SSL_352', help='Configuration name')
    parser.add_argument('--split', default='test', help='Dataset split to analyze')
    parser.add_argument('--max_samples', type=int, help='Maximum samples to analyze')
    parser.add_argument('--save_dir', default='ssl_analysis', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Run analysis
    analyzer = SSLResultsAnalyzer(args.model_path, args.config)
    results = analyzer.analyze_all(
        split=args.split,
        max_samples=args.max_samples,
        save_dir=args.save_dir
    ) 