from experiment import *
import os

def ddpm():
    """
    base configuration for all DDIM-based models.
    """
    conf = TrainConfig()
    conf.batch_size = 16 
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq'
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_ddpm
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 128 
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf

def autoenc_base():
    """
    base configuration for all Diff-AE models.
    """
    conf = TrainConfig()
    conf.batch_size = 16 
    conf.beatgans_gen_type = GenerativeType.ddim
    conf.beta_scheduler = 'linear'
    conf.data_name = 'ffhq' 
    conf.diffusion_type = 'beatgans'
    conf.eval_ema_every_samples = 200_000
    conf.eval_every_samples = 200_000
    conf.fp16 = True
    conf.lr = 1e-4
    conf.model_name = ModelName.beatgans_autoenc
    conf.net_attn = (16, )
    conf.net_beatgans_attn_head = 1
    conf.net_beatgans_embed_channels = 512 
    conf.net_beatgans_resnet_two_cond = True
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_ch = 64
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_pool = 'adaptivenonzero'
    conf.sample_size = 32
    conf.T_eval = 20
    conf.T = 1000
    conf.make_model_conf()
    return conf


def Example_autoenc_base():
    conf = autoenc_base() 
    conf.data_name = 'Example'  
    conf.scale_up_gpus(1)
    conf.img_size = 128
    conf.net_ch = 32     
    # final resolution = 8x8
    conf.net_ch_mult = (1, 2, 3, 4)
    # final resolution = 4x4
    conf.net_enc_channel_mult = (1, 2, 3, 4)
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.make_model_conf() 
    conf.contrast = True 
    return conf 


def Example_autoenc():
    conf = Example_autoenc_base() 
    conf.total_samples = 130_000_000
    conf.eval_ema_every_samples = 10_000_000
    conf.eval_every_samples = 10_000_000
    conf.name = 'Example_autoenc'
    return conf 


def VinDR_SSL_autoenc_base():
    """
    Base configuration for self-supervised learning on VinDR chest X-rays
    """
    conf = autoenc_base()
    conf.data_name = 'VinDR'
    conf.img_size = 352  # Higher resolution for chest X-rays
    conf.batch_size = 8  # Smaller batch size due to higher resolution
    conf.net_ch = 64
    
    # Self-supervised learning settings
    conf.self_supervised = True
    conf.augmentation_strength = 0.3  # REDUCED from 0.8 - critical for medical imaging
    conf.use_pathology_labels_for_training = False  # Fully self-supervised
    conf.validate_against_labels = True  # Use labels only for validation metrics
    
    # Visualization settings
    conf.compute_tsne = True
    conf.save_embeddings = True
    conf.tsne_perplexity = 30
    
    # Improved contrastive learning parameters
    conf.alpha = 2.0  # INCREASED contrastive loss weight
    conf.load_in = 5   # REDUCED - start contrastive learning earlier
    conf.K = 5         # Use fewer neighbors for prediction loss
    
    # Higher resolution architecture
    conf.net_ch_mult = (1, 2, 4, 8)
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8)
    conf.net_enc_pool = 'adaptivenonzero'
    
    # Evaluation settings - disable frequent sampling to avoid crashes
    conf.eval_ema_every_samples = 1_000_000  # Very infrequent
    conf.eval_every_samples = 1_000_000  # Very infrequent
    conf.sample_every_samples = 1_000_000  # Disable sampling during training
    
    conf.make_model_conf()
    return conf


def VinDR_SSL_352():
    """
    VinDR self-supervised learning at 352x352 resolution
    """
    conf = VinDR_SSL_autoenc_base()
    conf.img_size = 352
    conf.batch_size = 8
    conf.total_samples = 1_000_000
    conf.name = 'VinDR_SSL_352'
    # Use current directory structure
    conf.vindr_data_path = os.path.abspath('.')  # Current directory contains data/ folder
    conf.vindr_split = 'train'
    
    # Use smaller model to avoid GroupNorm issues
    conf.net_ch = 32  # Reduce base channels
    conf.net_ch_mult = (1, 2, 4)  # Simpler architecture
    conf.net_enc_channel_mult = (1, 2, 4)  # Simpler encoder
    conf.make_model_conf()
    
    return conf


def VinDR_SSL_512():
    """
    VinDR self-supervised learning at 512x512 resolution
    """
    conf = VinDR_SSL_autoenc_base()
    conf.img_size = 512
    conf.batch_size = 4  # Even smaller batch size for 512x512
    conf.net_ch = 32  # Smaller model to fit in memory
    conf.total_samples = 1_000_000
    conf.name = 'VinDR_SSL_512'
    # Use current directory structure
    conf.vindr_data_path = os.path.abspath('.')  # Current directory contains data/ folder
    conf.vindr_split = 'train'
    
    # Adjust model architecture for higher resolution
    conf.net_ch_mult = (1, 2, 4, 8, 8)
    conf.net_enc_channel_mult = (1, 2, 4, 8, 8, 8)
    conf.make_model_conf()
    return conf


def VinDR_SSL_validation():
    """
    Configuration for validation/analysis on VinDR test set
    """
    conf = VinDR_SSL_352()
    conf.vindr_split = 'test'
    conf.name = 'VinDR_SSL_validation'
    conf.validate_against_labels = True
    conf.compute_tsne = True
    conf.save_embeddings = True
    return conf

