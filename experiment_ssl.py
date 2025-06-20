import copy
import json
import os
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import *
from torch import nn
from torch.cuda import amp
from torch.optim.optimizer import Optimizer
from torch.utils.data.dataset import TensorDataset
from torchvision.utils import make_grid, save_image
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F

from config import *
from dataset import *
from dist_utils import *
from experiment import LitModel, ema
from metrics import *
from renderer import *

torch.backends.cudnn.enabled = False
import random


class SelfSupervisedLitModel(LitModel):
    """
    Self-supervised contrastive learning model for medical image phenotype discovery.
    Inherits from the base LitModel but overrides training for self-supervised learning.
    """
    
    def __init__(self, conf: TrainConfig):
        super().__init__(conf)
        
        # Additional buffers for storing embeddings and analysis
        self.register_buffer('stored_embeddings', torch.empty(0))
        self.register_buffer('stored_labels', torch.empty(0))
        self.register_buffer('stored_indices', torch.empty(0))
        
        # Validation tracking
        self.validation_embeddings = []
        self.validation_labels = []
        self.validation_pathologies = []
        
    def training_step(self, batch, batch_idx):
        """
        Self-supervised training step using contrastive learning on augmented pairs
        """
        with amp.autocast(False):
            # Extract images and augmented versions
            if self.conf.self_supervised and 'img_augmented' in batch:
                imgs = batch['img']
                imgs_aug = batch['img_augmented']
                x_start = imgs
                x_start_aug = imgs_aug
            else:
                # Fallback to standard training
                imgs = batch['img']
                x_start = imgs
                x_start_aug = None
            
            # Standard diffusion training
            t, weight = self.T_sampler.sample(len(x_start), x_start.device)
            
            # Prepare model kwargs for augmented pairs
            model_kwargs = {}
            if x_start_aug is not None:
                model_kwargs['x_start_augmented'] = x_start_aug
            
            # Get diffusion losses with self-supervised contrastive learning
            losses = self.sampler.training_losses(
                model=self.model,
                x_start=x_start,
                t=t,
                labels=None,  # No labels for self-supervised learning
                K=self.conf.K,
                alpha=self.conf.alpha,
                epsilon=self.conf.epsilon,
                current_epoch=self.current_epoch,
                load_in=self.conf.load_in,
                model_kwargs=model_kwargs
            )
            
            # L1 regularization on encoder
            normie = 0.00003 * torch.norm(
                torch.cat([param.data.view(-1) for (name, param) in self.model.named_parameters() 
                          if 'encoder' in name]), p=1
            )
            
            # Compute total loss
            if self.current_epoch > self.conf.load_in:
                loss = losses['mse'].mean() + normie
                if "contrastive-loss" in losses:
                    loss += losses['contrastive-loss']
                if "pred-loss" in losses:
                    loss += losses['pred-loss']
            else:
                loss = (losses['mse'].mean() * 100) + normie
            
            # Logging
            if self.global_rank == 0:
                self.logger.experiment.add_scalar('total_loss', loss, self.num_samples)
                self.logger.experiment.add_scalar('MSE', losses['mse'].mean(), self.num_samples)
                self.logger.experiment.add_scalar('L1_Norm', normie, self.num_samples)
                
                if "contrastive-loss" in losses:
                    self.logger.experiment.add_scalar('contrastive_loss', 
                                                     losses['contrastive-loss'], self.num_samples)
                    
                self.logger.experiment.add_scalar('LR', 
                                                 self.optimizer.param_groups[0]['lr'], 
                                                 self.current_epoch)
            
            return {'loss': loss}
    
    def on_train_epoch_end(self) -> None:
        """
        At the end of each epoch, perform clustering analysis and t-SNE visualization
        """
        if self.current_epoch >= self.conf.load_in and self.current_epoch % 10 == 0:
            self.analyze_embeddings()
    
    def analyze_embeddings(self):
        """
        Extract embeddings, perform clustering, and create visualizations
        """
        if not self.conf.validate_against_labels:
            return
            
        # Get validation dataset with labels
        val_dataset = self.conf.make_dataset()
        val_dataset.return_labels = True
        val_dataset.self_supervised = False  # Don't need augmented pairs for analysis
        
        val_loader = self.conf.make_loader(
            val_dataset,
            shuffle=False,
            batch_size=self.conf.batch_size_eval,
            drop_last=False
        )
        
        embeddings_list = []
        pathology_labels_list = []
        dominant_pathology_list = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in val_loader:
                # Debug: Print batch structure
                if len(embeddings_list) == 0:  # Only print for first batch
                    print(f"Batch type: {type(batch)}")
                    if isinstance(batch, dict):
                        print(f"Batch keys: {batch.keys()}")
                        print(f"Type of batch['img']: {type(batch['img'])}")
                        if isinstance(batch['img'], (list, tuple)):
                            print(f"Length of batch['img']: {len(batch['img'])}")
                            print(f"Type of batch['img'][0]: {type(batch['img'][0])}")
                
                # Handle different batch structures
                if isinstance(batch, dict):
                    # Dictionary format - check if img is a tensor or list
                    if isinstance(batch['img'], (list, tuple)):
                        # If img is a list/tuple, take the first element
                        imgs = batch['img'][0].to(self.device)
                    else:
                        # If img is a tensor
                        imgs = batch['img'].to(self.device)
                    
                    if 'pathology_labels' in batch:
                        pathology_labels_list.append(batch['pathology_labels'])
                        dominant_pathology_list.append(batch['dominant_pathology'])
                elif isinstance(batch, (list, tuple)):
                    # List/tuple format - unpack
                    if len(batch) >= 3:
                        imgs = batch[0].to(self.device)
                        pathology_labels_list.append(batch[1])
                        dominant_pathology_list.append(batch[2])
                    else:
                        imgs = batch[0].to(self.device)
                else:
                    # Single tensor
                    imgs = batch.to(self.device)
                
                # Extract embeddings
                embeddings = self.model.encoder(imgs)
                embeddings_list.append(embeddings.cpu())
        
        if len(embeddings_list) == 0:
            return
            
        # Concatenate all embeddings
        all_embeddings = torch.cat(embeddings_list, dim=0).numpy()
        
        if len(pathology_labels_list) > 0:
            # Convert to tensors if they're not already
            pathology_tensors = []
            dominant_tensors = []
            
            for labels in pathology_labels_list:
                if isinstance(labels, torch.Tensor):
                    pathology_tensors.append(labels.cpu())
                else:
                    pathology_tensors.append(torch.tensor(labels))
                    
            for labels in dominant_pathology_list:
                if isinstance(labels, torch.Tensor):
                    dominant_tensors.append(labels.cpu())
                else:
                    dominant_tensors.append(torch.tensor(labels))
            
            all_pathology_labels = torch.cat(pathology_tensors, dim=0).numpy()
            all_dominant_pathology = torch.cat(dominant_tensors, dim=0).numpy()
            
            # Perform clustering analysis
            self.clustering_analysis(all_embeddings, all_pathology_labels, all_dominant_pathology)
            
            # Create t-SNE visualization
            if self.conf.compute_tsne:
                self.tsne_visualization(all_embeddings, all_dominant_pathology)
        
        self.model.train()
    
    def clustering_analysis(self, embeddings, pathology_labels, dominant_pathology):
        """
        Perform clustering analysis and compute metrics
        """
        # K-means clustering
        n_clusters = len(self.conf.vindr_pathologies)  # 14 pathologies + no finding
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Compute clustering metrics
        ari = adjusted_rand_score(dominant_pathology, cluster_labels)
        nmi = normalized_mutual_info_score(dominant_pathology, cluster_labels)
        
        # Log metrics
        if self.global_rank == 0:
            self.logger.experiment.add_scalar('clustering/ARI', ari, self.current_epoch)
            self.logger.experiment.add_scalar('clustering/NMI', nmi, self.current_epoch)
            
            print(f"Epoch {self.current_epoch} - Clustering Metrics:")
            print(f"  Adjusted Rand Index: {ari:.4f}")
            print(f"  Normalized Mutual Information: {nmi:.4f}")
    
    def tsne_visualization(self, embeddings, dominant_pathology):
        """
        Create t-SNE visualization of embeddings colored by pathology
        """
        if self.global_rank != 0:  # Only create plots on main process
            return
            
        try:
            # Subsample for t-SNE if too many points
            max_points = 2000
            if len(embeddings) > max_points:
                indices = np.random.choice(len(embeddings), max_points, replace=False)
                embeddings_tsne = embeddings[indices]
                pathology_tsne = dominant_pathology[indices]
            else:
                embeddings_tsne = embeddings
                pathology_tsne = dominant_pathology
            
            # Compute t-SNE
            tsne = TSNE(n_components=2, perplexity=self.conf.tsne_perplexity, 
                       random_state=42, n_iter=1000)
            embeddings_2d = tsne.fit_transform(embeddings_tsne)
            
            # Create plot
            plt.figure(figsize=(12, 10))
            
            # Get pathology names
            pathology_names = self.conf.vindr_pathologies + ('No Finding',)
            colors = plt.cm.tab20(np.linspace(0, 1, len(pathology_names)))
            
            for i, (pathology, color) in enumerate(zip(pathology_names, colors)):
                mask = pathology_tsne == i
                if mask.sum() > 0:
                    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                              c=[color], label=pathology, alpha=0.7, s=20)
            
            plt.title(f't-SNE Visualization of Learned Embeddings (Epoch {self.current_epoch})')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(self.conf.logdir, 'tsne_plots')
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, f'tsne_epoch_{self.current_epoch}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log to tensorboard
            # Convert plot to image and log
            plt.figure(figsize=(12, 10))
            for i, (pathology, color) in enumerate(zip(pathology_names, colors)):
                mask = pathology_tsne == i
                if mask.sum() > 0:
                    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                              c=[color], label=pathology, alpha=0.7, s=20)
            plt.title(f't-SNE Visualization (Epoch {self.current_epoch})')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            
            # Convert to tensor for logging
            plt.tight_layout()
            plt.savefig('temp_tsne.png', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Load and log the image
            img = plt.imread('temp_tsne.png')
            img_tensor = torch.from_numpy(img).permute(2, 0, 1)
            self.logger.experiment.add_image('tsne/embedding_visualization', 
                                           img_tensor, self.current_epoch)
            
            # Clean up
            if os.path.exists('temp_tsne.png'):
                os.remove('temp_tsne.png')
                
        except Exception as e:
            print(f"Error creating t-SNE visualization: {e}")
    
    def extract_prototypes(self, n_prototypes=5):
        """
        Extract prototype images for each discovered cluster
        """
        # This would be implemented to find the most representative images
        # for each cluster center
        pass
    
    def save_embeddings_for_analysis(self, save_path):
        """
        Save embeddings and labels for external analysis
        """
        if not self.conf.save_embeddings:
            return
            
        # Implementation to save embeddings, labels, and metadata
        # for further analysis in notebooks or other tools
        pass


def train_self_supervised(conf: TrainConfig, gpus, nodes=1):
    """
    Training function specifically for self-supervised learning
    """
    print("Starting self-supervised training...")
    print(f"Configuration: {conf.name}")
    print(f"Self-supervised: {conf.self_supervised}")
    print(f"Image size: {conf.img_size}")
    print(f"Augmentation strength: {conf.augmentation_strength}")
    
    # Ensure self-supervised mode is enabled
    conf.self_supervised = True
    
    # Use the self-supervised model
    model = SelfSupervisedLitModel(conf)
    
    # Set up training
    if conf.seed is not None:
        pl.seed_everything(conf.seed)
    
    # Callbacks
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(conf.logdir, 'checkpoints'),
        filename='ssl-{epoch:02d}-{total_loss:.4f}',
        monitor='total_loss',
        mode='min',
        save_top_k=3,
        save_last=True,
        every_n_epochs=10
    )
    callbacks.append(checkpoint_callback)
    
    # Logger
    logger = pl_loggers.TensorBoardLogger(
        save_dir=conf.base_dir,
        name=conf.name,
        version=''
    )
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=conf.num_epochs,
        gpus=gpus,
        num_nodes=nodes,
        accelerator='ddp' if len(gpus) > 1 else None,
        precision=16 if conf.fp16 else 32,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=conf.grad_clip,
        accumulate_grad_batches=conf.accum_batches,
        check_val_every_n_epoch=10,
        log_every_n_steps=100
    )
    
    # Start training
    trainer.fit(model)
    
    print("Self-supervised training completed!")
    return model, trainer 