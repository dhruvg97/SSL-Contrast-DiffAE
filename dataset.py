import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
import pydicom
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
import random
import nibabel as nib

class MedicalImageTransforms:
    """Medical image augmentations for self-supervised contrastive learning"""
    
    def __init__(self, strength=0.8, img_size=352):
        self.img_size = img_size
        
        # Base transforms (minimal, for "positive" pairs)
        self.base_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            # Normalize to standard range for grayscale (1 channel)
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        # Conservative medical augmentations that preserve pathological features
        self.strong_transform = transforms.Compose([
            # Minimal resize crop instead of random crop
            transforms.Resize((int(img_size * 1.05), int(img_size * 1.05))),
            transforms.CenterCrop((img_size, img_size)),
            # Very conservative rotation (pathologies are orientation-sensitive)
            transforms.RandomAffine(
                degrees=3 * strength,  # Max 2.4° instead of 12°
                translate=(0.02 * strength, 0.02 * strength),  # Minimal translation
                scale=(1 - 0.02 * strength, 1 + 0.02 * strength)  # Minimal scaling
            ),
            # NO horizontal flip for chest X-rays (anatomical orientation matters)
            # Medical-appropriate intensity changes
            transforms.ColorJitter(
                brightness=0.1 * strength,  # Reduced from 0.16
                contrast=0.15 * strength    # Reduced from 0.16
            ),
            transforms.ToTensor(),
            # Much reduced noise to preserve fine details
            transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.005 * strength),
            transforms.Lambda(lambda x: torch.clamp(x, 0, 1)),
            # Normalize to standard range
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __call__(self, image):
        """Returns original and augmented version for contrastive learning"""
        return self.base_transform(image), self.strong_transform(image)


class VinDRChestXrayDataset(Dataset):
    """VinDR Chest X-ray dataset for self-supervised contrastive learning"""
    
    def __init__(self, 
                 data_path,
                 split='train',
                 img_size=352,
                 self_supervised=True,
                 augmentation_strength=0.8,
                 return_labels=False):
        """
        Args:
            data_path: Path to VinDR dataset root (contains data/ folder)
            split: 'train', 'val', or 'test'
            img_size: Target image size (352 or 512)
            self_supervised: Whether to return augmented pairs
            augmentation_strength: Strength of augmentations (0.0 to 1.0)
            return_labels: Whether to return pathology labels (for validation)
        """
        self.data_path = Path(data_path)
        self.split = split
        self.img_size = img_size
        self.self_supervised = self_supervised
        self.return_labels = return_labels
        
        # Load CSV annotations
        csv_path = self.data_path / 'data' / f'{split}.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        print(f"Loading annotations from: {csv_path}")
        self.annotations = pd.read_csv(csv_path)
        
        # Get unique image IDs (since CSV contains multiple rows per image for different findings)
        self.unique_images = self.annotations['image_id'].unique()
        print(f"Found {len(self.unique_images)} unique images in {split} split")
        
        # Path to DICOM images
        self.images_dir = self.data_path / 'data' / split
        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        
        # Verify some images exist
        sample_image_path = self.images_dir / f"{self.unique_images[0]}.dicom"
        if not sample_image_path.exists():
            raise FileNotFoundError(f"Sample image not found: {sample_image_path}")
        
        # VinDR pathology class mapping (based on class_id in CSV)
        self.class_names = {
            0: 'Aortic enlargement',
            1: 'Atelectasis', 
            2: 'Calcification',
            3: 'Cardiomegaly',
            4: 'Consolidation',
            5: 'ILD',
            6: 'Infiltration',
            7: 'Lung Opacity',
            8: 'Nodule/Mass',
            9: 'Other lesion',
            10: 'Pleural effusion',
            11: 'Pleural thickening',
            12: 'Pneumothorax',
            13: 'Pulmonary fibrosis',
            14: 'No finding'
        }
        
        # Set up transforms
        if self_supervised:
            self.transform = MedicalImageTransforms(augmentation_strength, img_size)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        
        print(f"Dataset initialized: {len(self)} images, SSL={self_supervised}, labels={return_labels}")
        
    def __len__(self):
        return len(self.unique_images)
    
    def _load_dicom_image(self, image_id):
        """Load and preprocess DICOM image"""
        dicom_path = self.images_dir / f"{image_id}.dicom"
        
        try:
            # Load DICOM file
            dicom_data = pydicom.dcmread(dicom_path)
            image_array = dicom_data.pixel_array
            
            # Handle different DICOM formats
            if len(image_array.shape) == 3:
                # Multi-frame DICOM, take first frame
                image_array = image_array[0]
            
            # Normalize to 0-1 range
            image_array = image_array.astype(np.float32)
            if image_array.max() > 1:
                image_array = image_array / image_array.max()
            
            # Convert to PIL Image for transforms
            # Convert to 8-bit for PIL
            image_array = (image_array * 255).astype(np.uint8)
            image = Image.fromarray(image_array).convert('L')
            
            return image
            
        except Exception as e:
            print(f"Error loading DICOM {dicom_path}: {e}")
            # Return a blank image as fallback
            return Image.new('L', (512, 512), 0)
    
    def __getitem__(self, idx):
        # Get image ID
        image_id = self.unique_images[idx]
        
        # Load DICOM image
        image = self._load_dicom_image(image_id)
        
        sample = {
            'index': idx,
            'image_id': image_id,
        }
        
        if self.self_supervised:
            # Return original and augmented versions for contrastive learning
            img_original, img_augmented = self.transform(image)
            sample['img'] = img_original
            sample['img_augmented'] = img_augmented
        else:
            sample['img'] = self.transform(image)
        
        if self.return_labels:
            # Get all annotations for this image
            image_annotations = self.annotations[self.annotations['image_id'] == image_id]
            
            # Create binary labels for each pathology class
            pathology_labels = np.zeros(15, dtype=np.float32)  # 15 classes (0-14)
            
            for _, row in image_annotations.iterrows():
                class_id = int(row['class_id'])
                pathology_labels[class_id] = 1.0
            
            sample['pathology_labels'] = torch.tensor(pathology_labels)
            
            # Create binary label (any pathology present, excluding "No finding")
            has_pathology = float(pathology_labels[:14].sum() > 0)  # Exclude class 14 (No finding)
            sample['binary_label'] = torch.tensor(has_pathology)
            
            # Get dominant pathology for clustering analysis
            if has_pathology:
                # Find the first pathology (excluding "No finding")
                dominant_pathology = np.argmax(pathology_labels[:14])
            else:
                dominant_pathology = 14  # "No finding" class
            sample['dominant_pathology'] = torch.tensor(dominant_pathology)
        
        return sample
    
    def get_pathology_names(self):
        """Return list of pathology names"""
        return [self.class_names[i] for i in range(15)]

# load the slices
class ExampleDataset(Dataset):
    """ SliceLoader 
    
    A class which is used to allow for efficient data loading of the training data. 
    Args:
        - torch.utils.data.Dataset: A PyTorch module from which this class inherits which allows it 
        to make use of the PyTorch dataloader functionalities. 
    
    """
    def __init__(self, train=False, val=False, test=False, epoch_end=True, N=4632):
        #4632
        """ Class constructor
        
        Args:
            - downsampling_factor: The factor by which the loaded data has been downsampled. 
            - N: The length of the dataset. 
            - folder_name: The folder from which the data comes from 
            - is_train: Whether or not the dataloader is loading training data (and therefore randomised data).   
        """ 
        self.val = val 
        self.test = test 
        self.train = train
        self.N = N - 1
        self.data = pd.read_csv('brain_age.csv')
        self.train_data = self.data.iloc[:N] 
        self.val_data = self.data.iloc[4631:]
        self.epoch_end = epoch_end
        
        
    def __len__(self):
        """ __len__
        
        A function which configures and returns the size of the datset. 
        
        Output: 
            - N: The size of the dataset. 
        """
        return (self.N)    
    
    def __getitem__(self, idx):
        transforms_list = [transforms.ToPILImage(), 
                          transforms.RandomAffine(degrees=(0, 350), translate=(0.1, 0.12)), transforms.Resize((64, 64)),
                          transforms.ToTensor()]
        if self.train: 
            #get the filepath
            file_path = self.train_data['file_path'].iloc[idx] 
            #get the root directory of file path
            dataset = file_path.split('/')[0] 
            #load the image
            image = self._load_nib(file_path)[0, :, :, 72] if dataset == 'NACC' else self._load_nib(file_path) 
            #load the label 
            label = random.choice([0, 1])  
            #apply the transforms 
            image = transforms.Compose(transforms_list)(image) if not self.epoch_end else transforms.Compose([transforms_list[0], transforms_list[-2], transforms_list[-1]])(image)

        elif self.val: 
            #get the filepath
            file_path = self.val_data['file_path'].iloc[idx] 
            #get the root directory of file path
            dataset = file_path.split('/')[0] 
            #load the image
            image = self._load_nib(file_path)[0, :, :, 72] if dataset == 'NACC' else self._load_nib(file_path)
            #load the label 
            label = random.choice([0, 1])   
            #apply the transforms 
            image = transforms.Compose([transforms_list[0], transforms_list[-2], transforms_list[-1]])(image)
        

        return {'img': image, 'index': idx, 'label': label, 'file_path': file_path}    
    
    
    def _load_nib(self, filename): 
        """ _load_nib 
        
        A function to load compressed nifti images.
        Args:
            - filename: The name of the file to be loaded. 
        Ouput:
            - The corresponding image as a PyTorch tensor. 
        
        """
        return torch.tensor(nib.load(filename).get_fdata(), dtype=torch.float) 

