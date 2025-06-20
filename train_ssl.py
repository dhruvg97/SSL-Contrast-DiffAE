#!/usr/bin/env python3

"""
Self-Supervised Learning Training Script for VinDR Chest X-ray Dataset
Optimized for Multi-GPU Training
"""

import os
import sys
import torch
import argparse
from pathlib import Path
import numpy as np

# Add current directory to path for imports
sys.path.append(os.getcwd())

from templates import VinDR_SSL_352, VinDR_SSL_512
from experiment_ssl import SelfSupervisedLitModel
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor


def main():
    # Parse command line arguments for input
    parser = argparse.ArgumentParser(description='Train SSL Contrast-DiffAE on VinDR dataset')
    parser.add_argument('--resolution', type=int, choices=[352, 512], default=352,
                       help='Image resolution (352 or 512)')
    parser.add_argument('--gpus', type=int, default=2,
                       help='Number of GPUs to use (1 or 2)')
    parser.add_argument('--max_epochs', type=int, default=100,
                       help='Maximum number of epochs')
    parser.add_argument('--batch_size_per_gpu', type=int, default=None,
                       help='Batch size per GPU (auto-calculated if not specified)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--strategy', type=str, default='ddp', choices=['ddp', 'dp'],
                       help='Multi-GPU strategy: ddp (recommended) or dp')
    
    args = parser.parse_args()
    
    # Check if data exists
    data_dir = Path('./data')
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    train_csv = data_dir / 'train.csv'
    train_images = data_dir / 'train'
    
    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")
    
    if not train_images.exists():
        raise FileNotFoundError(f"Training images directory not found: {train_images}")
    
    print(f"✓ Found data directory: {data_dir}")
    print(f"✓ Found training CSV: {train_csv}")
    print(f"✓ Found training images: {train_images}")
    
    # GPU validation
    available_gpus = torch.cuda.device_count()
    if available_gpus == 0:
        raise RuntimeError("No GPUs available!")
    
    if args.gpus > available_gpus:
        print(f"⚠️  Requested {args.gpus} GPUs but only {available_gpus} available. Using {available_gpus}.")
        args.gpus = available_gpus
    
    print(f"🔥 Using {args.gpus} GPU(s) out of {available_gpus} available")
    
    # Load configuration based on resolution
    if args.resolution == 352:
        config = VinDR_SSL_352()
        print("Using 352x352 resolution configuration")
    else:
        config = VinDR_SSL_512()
        print("Using 512x512 resolution configuration")
    
    # Override some settings for your specific setup
    config.vindr_data_path = str(Path.cwd())  # Current directory contains data/ folder
    config.num_epochs = args.max_epochs
    config.total_samples = args.max_epochs * 1000  # Adjust based on your dataset size
    
    # Optimize batch size for multi-GPU training
    if args.batch_size_per_gpu is not None:
        base_batch_size = args.batch_size_per_gpu
    else:
        # OPTIMIZED batch sizes for A5000 GPUs (24GB each) - Much higher utilization
        if args.resolution == 352:
            base_batch_size = 20  # per GPU - total batch size scales with GPU count
        else:  # 512x512
            base_batch_size = 12  # per GPU - total batch size scales with GPU count
    
    # Set batch size per GPU (effective batch size will be base_batch_size * gpus)
    config.batch_size = base_batch_size
    config.batch_size_eval = base_batch_size
    
    # Scale learning rate with effective batch size for multi-GPU training
    # Use square root scaling which is more stable than linear scaling
    if args.gpus > 1:
        # Calculate effective batch size scaling factor
        # Square root scaling: more conservative than linear scaling  
        lr_scale_factor = np.sqrt(args.gpus)
        effective_batch_size = base_batch_size * args.gpus
        
        config.lr = config.lr * lr_scale_factor
        print(f"📈 Scaled learning rate to {config.lr:.6f} for {args.gpus} GPUs")
        print(f"    Effective batch size: {effective_batch_size} (scale factor: {lr_scale_factor:.2f})")
        print(f"    Using square root scaling for stability")
    
    print(f"\n🔧 Multi-GPU Configuration:")
    print(f"  - GPUs: {args.gpus}")
    print(f"  - Strategy: {args.strategy.upper()}")
    print(f"  - Batch size per GPU: {base_batch_size}")
    print(f"  - Effective batch size: {effective_batch_size}")
    print(f"  - Learning rate: {config.lr:.6f}")
    print(f"  - Data path: {config.vindr_data_path}")
    print(f"  - Image size: {config.img_size}")
    print(f"  - Self-supervised: {config.self_supervised}")
    print(f"  - Augmentation strength: {config.augmentation_strength}")
    print(f"  - Max epochs: {args.max_epochs}")
    
    # Create model
    print("\n🚀 Initializing SSL Contrast-DiffAE model...")
    model = SelfSupervisedLitModel(config)
    
    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoints/{config.name}',
        filename='{epoch:02d}-{step}-{val_loss:.3f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min',
        save_last=True,
        every_n_epochs=5,  # Save more frequently for long training
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # GPU monitoring callback
    device_monitor = DeviceStatsMonitor()
    
    callbacks = [checkpoint_callback, lr_monitor, device_monitor]
    
    # Set up trainer with optimized multi-GPU settings
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        devices=args.gpus,
        accelerator='gpu',
        strategy=args.strategy,
        callbacks=callbacks,
        precision=16 if config.fp16 else 32,
        gradient_clip_val=config.grad_clip,
        accumulate_grad_batches=config.accum_batches,
        log_every_n_steps=25,
        val_check_interval=0.25,  # Validate 4 times per epoch
        sync_batchnorm=True if args.gpus > 1 else False,
        enable_progress_bar=True,
        enable_model_summary=True,
        detect_anomaly=False,  # Disable for speed
        benchmark=True,  # Enable for consistent input sizes
    )
    
    print(f"\n🏁 Starting multi-GPU SSL training for {args.max_epochs} epochs...")
    print(f"📁 Checkpoints will be saved to: checkpoints/{config.name}/")
    print(f"📊 Logs will be saved to: lightning_logs/")
    print(f"🎯 Strategy: {args.strategy.upper()} (Distributed Data Parallel)")
    
    if args.gpus > 1:
        print(f"\n💡 Multi-GPU Tips:")
        print(f"  - Monitor GPU utilization with: watch -n 1 nvidia-smi")
        print(f"  - Each GPU will process {base_batch_size} samples per batch")
        print(f"  - Gradients are synchronized across GPUs every step")
        print(f"  - Model replicated on each GPU for maximum efficiency")
    
    # Start training
    try:
        trainer.fit(model, ckpt_path=args.resume_from_checkpoint) #main training loop
        print("\n🎉 Training completed successfully!")
        print(f"💾 Best checkpoint: {checkpoint_callback.best_model_path}")
        
        # Final GPU memory report
        if torch.cuda.is_available():
            print(f"\n📊 Final GPU Memory Usage:")
            for i in range(args.gpus):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                cached = torch.cuda.memory_reserved(i) / 1024**3
                print(f"  GPU {i}: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
                
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


if __name__ == '__main__':
    main() 