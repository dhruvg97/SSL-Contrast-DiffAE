#!/usr/bin/env python3

"""
Analysis script for t-SNE results from SSL training
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

# Add current directory to path for imports
sys.path.append(os.getcwd())

from templates import VinDR_SSL_352
from dataset import VinDRChestXrayDataset

def analyze_dataset_distribution():
    """Analyze the class distribution in the VinDR dataset"""
    
    config = VinDR_SSL_352()
    config.vindr_data_path = str(Path.cwd())
    
    # Load dataset with labels
    dataset = VinDRChestXrayDataset(
        data_path=config.vindr_data_path,
        split='train',
        img_size=config.img_size,
        self_supervised=False,
        return_labels=True
    )
    
    print("=== VinDR Dataset Analysis ===")
    print(f"Total images: {len(dataset)}")
    print()
    
    # Count dominant pathologies
    dominant_counts = Counter()
    pathology_counts = Counter()
    
    # Sample a subset for analysis (to avoid processing all 15k images)
    sample_size = min(1000, len(dataset))
    indices = np.random.choice(len(dataset), sample_size, replace=False)
    
    print(f"Analyzing {sample_size} random samples...")
    
    for idx in indices:
        sample = dataset[idx]
        
        if 'dominant_pathology' in sample:
            dominant_pathology = sample['dominant_pathology'].item()
            dominant_counts[dominant_pathology] += 1
        
        if 'pathology_labels' in sample:
            pathology_labels = sample['pathology_labels'].numpy()
            for i, label in enumerate(pathology_labels):
                if label > 0:
                    pathology_counts[i] += 1
    
    # Map class IDs to names
    class_names = dataset.class_names
    
    print("\n=== Dominant Pathology Distribution ===")
    total_samples = sum(dominant_counts.values())
    
    for class_id, count in sorted(dominant_counts.items()):
        percentage = (count / total_samples) * 100
        class_name = class_names.get(class_id, f"Unknown_{class_id}")
        print(f"{class_name:20s}: {count:4d} ({percentage:5.1f}%)")
    
    print(f"\nTotal samples analyzed: {total_samples}")
    
    # Calculate expected behavior for t-SNE
    print("\n=== Expected t-SNE Behavior ===")
    
    no_finding_percentage = (dominant_counts.get(14, 0) / total_samples) * 100
    pathological_percentage = 100 - no_finding_percentage
    
    print(f"• {no_finding_percentage:.1f}% of images show 'No Finding'")
    print(f"• {pathological_percentage:.1f}% of images have pathological findings")
    print()
    
    if no_finding_percentage > 60:
        print("⚠️  High percentage of 'No Finding' cases detected!")
        print("   This is common in chest X-ray datasets and explains why")
        print("   most points in your t-SNE cluster around 'No Finding'.")
    
    print("\n=== What to Look for in t-SNE ===")
    print("1. **Dominant Cluster**: Large cluster of 'No Finding' cases (expected)")
    print("2. **Pathology Separation**: Distinct clusters for different pathologies")
    print("3. **Similar Pathologies**: Related conditions should cluster nearby")
    print("   - Atelectasis, Consolidation, Infiltration (lung conditions)")
    print("   - Cardiomegaly, Aortic enlargement (cardiac conditions)")
    print("   - Pleural effusion, Pleural thickening (pleural conditions)")
    print("4. **Progression**: Clusters should become more distinct over epochs")
    
    return dominant_counts, pathology_counts, class_names

def create_analysis_summary():
    """Create a comprehensive analysis summary"""
    
    print("\n" + "="*60)
    print("t-SNE INTERPRETATION GUIDE")
    print("="*60)
    
    print("\n📊 WHAT YOUR t-SNE PLOT SHOULD SHOW:")
    print("┌" + "─"*58 + "┐")
    print("│ GOOD SIGNS:                                          │")
    print("│ • Clear separation between pathology clusters        │")
    print("│ • 'No Finding' forms a distinct, large cluster      │")
    print("│ • Similar pathologies cluster near each other       │")
    print("│ • Clusters become tighter over training epochs      │")
    print("│                                                      │")
    print("│ CONCERNING SIGNS:                                    │")
    print("│ • All points mixed together (poor learning)         │")
    print("│ • No clear 'No Finding' cluster                     │")
    print("│ • Random scattered points                           │")
    print("│ • No improvement from epoch 10 to 40               │")
    print("└" + "─"*58 + "┘")
    
    print("\n🔍 SPECIFIC PATHOLOGY GROUPS TO WATCH:")
    
    pathology_groups = {
        "Lung Parenchymal": ["Atelectasis", "Consolidation", "Infiltration", "ILD", "Pulmonary fibrosis"],
        "Pleural": ["Pleural effusion", "Pleural thickening", "Pneumothorax"],
        "Cardiac": ["Cardiomegaly", "Aortic enlargement"],
        "Masses/Nodules": ["Nodule/Mass", "Other lesion"],
        "Other": ["Calcification", "Lung Opacity"]
    }
    
    for group, pathologies in pathology_groups.items():
        print(f"\n{group} conditions:")
        for pathology in pathologies:
            print(f"  • {pathology}")
        print(f"  → Should cluster relatively close together")
    
    print(f"\n'No Finding' cases:")
    print(f"  • Should form the largest, most coherent cluster")
    print(f"  • Often 60-80% of chest X-ray datasets")
    
    print("\n📈 EPOCH PROGRESSION ANALYSIS:")
    print("• Epoch 10-20: Initial structure formation")
    print("• Epoch 20-30: Cluster refinement") 
    print("• Epoch 30-40: Fine-tuning and stabilization")
    print("• Look for: Tighter clusters, better separation over time")

def check_tensorboard_logs():
    """Check if there are tensorboard logs with clustering metrics"""
    
    print("\n" + "="*60)
    print("QUANTITATIVE METRICS")
    print("="*60)
    
    logs_dir = Path("lightning_logs")
    if logs_dir.exists():
        print("📊 Checking for quantitative clustering metrics...")
        
        # Find the latest version
        version_dirs = [d for d in logs_dir.iterdir() if d.is_dir() and d.name.startswith("version_")]
        if version_dirs:
            latest_version = max(version_dirs, key=lambda x: int(x.name.split("_")[1]))
            print(f"Latest log version: {latest_version.name}")
            
            events_files = list(latest_version.glob("events.out.tfevents.*"))
            if events_files:
                print(f"✓ Found tensorboard logs in {latest_version}")
                print("\nTo view clustering metrics, run:")
                print(f"tensorboard --logdir {latest_version}")
                print("\nLook for:")
                print("• clustering/ARI (Adjusted Rand Index)")
                print("• clustering/NMI (Normalized Mutual Information)")
                print("• Higher values = better clustering")
            else:
                print("⚠️  No tensorboard event files found")
        else:
            print("⚠️  No version directories found in lightning_logs")
    else:
        print("⚠️  No lightning_logs directory found")

def main():
    """Main analysis function"""
    
    print("🔬 SSL t-SNE Results Analysis")
    print("="*50)
    
    try:
        # Analyze dataset distribution
        dominant_counts, pathology_counts, class_names = analyze_dataset_distribution()
        
        # Create interpretation guide
        create_analysis_summary()
        
        # Check for quantitative metrics
        check_tensorboard_logs()
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("1. Open your t-SNE plots side by side (epochs 10, 20, 30, 40)")
        print("2. Look for the patterns described above")
        print("3. Run tensorboard to see quantitative metrics:")
        print("   tensorboard --logdir lightning_logs")
        print("4. If clustering is poor, consider:")
        print("   • Training for more epochs")
        print("   • Adjusting augmentation strength")
        print("   • Modifying contrastive loss parameters")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("Make sure you're in the correct directory with data/ folder")

if __name__ == "__main__":
    main() 