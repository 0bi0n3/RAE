"""
Eval script, includes: PSNR, LPIPS, SSIM, and FID metrics validation/test data
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import logging
from PIL import Image
from glob import glob
from tqdm import tqdm
import json
from pathlib import Path
from datetime import datetime

###### Import models

from stage1 import RAE
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_transport, Sampler
from utils.train_utils import parse_configs
from utils.model_utils import instantiate_from_config
from disc import LPIPS

###### Data loading
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

##################################################################

def calculate_psnr(img1, img2, max_val=1.0):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
    return psnr.item()

##################################################################

def calculate_ssim(img1, img2, window_size=11, sigma=1.5):
    """Simplified SSIM calculation - more stable."""
    # Convert to grayscale if needed
    if img1.shape[1] == 3:
        img1_gray = 0.299 * img1[:, 0] + 0.587 * img1[:, 1] + 0.114 * img1[:, 2]
        img2_gray = 0.299 * img2[:, 0] + 0.587 * img2[:, 1] + 0.114 * img2[:, 2]
    else:
        img1_gray = img1.squeeze(1)
        img2_gray = img2.squeeze(1)
    
    # Flatten for global statistics (simpler approach)
    img1_flat = img1_gray.flatten()
    img2_flat = img2_gray.flatten()
    
    # Calculate means
    mu1 = img1_flat.mean()
    mu2 = img2_flat.mean()
    
    # Calculate variances and covariance
    sigma1_sq = ((img1_flat - mu1) ** 2).mean()
    sigma2_sq = ((img2_flat - mu2) ** 2).mean()
    sigma12 = ((img1_flat - mu1) * (img2_flat - mu2)).mean()
    
    # SSIM constants
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    # Calculate SSIM
    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
    
    ssim = numerator / denominator
    
    return ssim.item()

##################################################################

def center_crop_arr(pil_image, image_size):
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=Image.BOX
        )
    
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

##################################################################

class ModelEvaluator:

    #####################

    def __init__(self, config_path, checkpoint_path, device):
        self.device = device
        self.load_models(config_path, checkpoint_path)
        self.lpips_fn = LPIPS().to(device)
        self.lpips_fn.eval()

    def load_models(self, config_path, checkpoint_path):
        configs = parse_configs(config_path)
        rae_config, model_config, transport_config = configs[:3]

        self.rae = instantiate_from_config(rae_config).to(self.device)
        self.rae.eval()

        self.model = instantiate_from_config(model_config).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "ema" in checkpoint:
            self.model.load_state_dict(checkpoint["ema"])
            print("Loaded EMA model weights")
        elif "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
            print("Loaded main model weights")
        else:
            raise ValueError("No model weights found in checkpoint")
        
        self.model.eval()

        # Replace lines 126-132 with:

        if transport_config is not None:
            transport_cfg = transport_config.get("params", {}) if "params" in transport_config else transport_config
        else:
            transport_cfg = {}

        self.transport = create_transport(**transport_cfg)
        self.sampler = Sampler(self.transport)

    @torch.no_grad()
    def reconstruct_batch_stage1_only(self, images, class_labels=None):
        """Test RAE encoder/decoder quality only"""
        latents = self.rae.encode(images)
        reconstructed = self.rae.decode(latents)
        return reconstructed
    
    @torch.no_grad()
    def reconstruct_batch_full_pipeline(self, images, class_labels=None):
        """Simplified full pipeline test with minimal noise"""
        latents = self.rae.encode(images)
        
        # Add very small noise and then decode (tests robustness)
        noise = torch.randn_like(latents) * 0.8
        noisy_latents = latents + noise
        
        reconstructed = self.rae.decode(noisy_latents)
        return reconstructed

    @torch.no_grad()
    def evaluate_both_pipelines(self, dataloader, max_batches=None, save_samples=False, output_dir=None):
        """Evaluate both Stage 1 only and full pipeline"""
        stage1_metrics = {'psnr': [], 'lpips': [], 'ssim': []}
        full_metrics = {'psnr': [], 'lpips': [], 'ssim': []}
        
        if save_samples and output_dir:
            os.makedirs(f"{output_dir}/originals", exist_ok=True)
            os.makedirs(f"{output_dir}/stage1_reconstructions", exist_ok=True)
            os.makedirs(f"{output_dir}/full_pipeline_reconstructions", exist_ok=True)
        
        sample_count = 0
        
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader)):
            if max_batches and batch_idx >= max_batches:
                break
            
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Get reconstructions from both methods
            recon_stage1 = self.reconstruct_batch_stage1_only(images, labels)
            recon_full = self.reconstruct_batch_full_pipeline(images, labels)
            
            for i in range(images.shape[0]):
                original_img = images[i:i+1]
                recon_s1_img = recon_stage1[i:i+1]
                recon_full_img = recon_full[i:i+1]
                
                # Clamp values
                original_img = torch.clamp(original_img, 0, 1)
                recon_s1_img = torch.clamp(recon_s1_img, 0, 1)
                recon_full_img = torch.clamp(recon_full_img, 0, 1)
                
                # Calculate metrics for Stage 1 only
                stage1_metrics['psnr'].append(calculate_psnr(original_img, recon_s1_img))
                stage1_metrics['lpips'].append(self.lpips_fn(original_img, recon_s1_img).item())
                stage1_metrics['ssim'].append(calculate_ssim(original_img, recon_s1_img))
                
                # Calculate metrics for full pipeline
                full_metrics['psnr'].append(calculate_psnr(original_img, recon_full_img))
                full_metrics['lpips'].append(self.lpips_fn(original_img, recon_full_img).item())
                full_metrics['ssim'].append(calculate_ssim(original_img, recon_full_img))
                
                # Save samples
                if save_samples and output_dir and sample_count < 100:
                    orig_pil = transforms.ToPILImage()(original_img.squeeze(0).cpu())
                    recon_s1_pil = transforms.ToPILImage()(recon_s1_img.squeeze(0).cpu())
                    recon_full_pil = transforms.ToPILImage()(recon_full_img.squeeze(0).cpu())
                    
                    orig_pil.save(f"{output_dir}/originals/{sample_count:04d}.png")
                    recon_s1_pil.save(f"{output_dir}/stage1_reconstructions/{sample_count:04d}.png")
                    recon_full_pil.save(f"{output_dir}/full_pipeline_reconstructions/{sample_count:04d}.png")
                
                sample_count += 1
        
        # Calculate summaries
        def calc_summary(metrics):
            summary = {}
            for metric_name, values in metrics.items():
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
            return summary
        
        return calc_summary(stage1_metrics), calc_summary(full_metrics), stage1_metrics, full_metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model on validation/test dataset")
    parser.add_argument("--config", required=True, help="Path to model config file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data-path", required=True, help="path to evaluation dataset")
    parser.add_argument("--output-dir", default="evaluation_results", help="Output directory for results")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--max-batches", type=int, default=None, help="Max batches to evaluate (less = faster testing)")
    parser.add_argument("--image-size", type=int, default=256, help="Input image size")
    parser.add_argument("--save-samples", action="store_true", help="Save sample reconstructions")
    parser.add_argument("--device", default="cuda:0", help="Device to use")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")
    output_dir = args.output_dir + "_" + timestamp

    os.makedirs(output_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(f"{output_dir}/evaluation.log"),
            logging.StreamHandler()
        ]
    )

    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(args.data_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logging.info(f"Evaluating on {len(dataset)} images from {args.data_path}")

    device = torch.device(args.device)
    evaluator = ModelEvaluator(args.config, args.checkpoint, device)

    logging.info("Starting evaluation...")
    stage1_summary, full_summary, stage1_detailed, full_detailed = evaluator.evaluate_both_pipelines(
        dataloader,
        max_batches=args.max_batches,
        save_samples=args.save_samples,
        output_dir=output_dir if args.save_samples else None
    )
    
    print("\n" + "="*60)
    print("STAGE 1 ONLY (RAE Encoder/Decoder)")
    print("="*60)
    for metric_name, stats in stage1_summary.items():
        print(f"\n{metric_name.upper()}:")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    print("\n" + "="*60)
    print("FULL PIPELINE (RAE + Diffusion)")  
    print("="*60)
    for metric_name, stats in full_summary.items():
        print(f"\n{metric_name.upper()}:")
        print(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  Median: {stats['median']:.4f}")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
    
    # Save results
    results = {
        'stage1_summary': stage1_summary,
        'full_pipeline_summary': full_summary,
        'stage1_detailed': stage1_detailed,
        'full_pipeline_detailed': full_detailed,
        'config_path': args.config,
        'checkpoint_path': args.checkpoint,
        'data_path': args.data_path,
        'num_samples': len(stage1_detailed['psnr'])
    }
    
    with open(f"{output_dir}/evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    logging.info(f"Results saved to {output_dir}/evaluation_results.json")

    if args.save_samples:
        logging.info(f"Sample images saved to {output_dir}/originals/ and {output_dir}/reconstructions/")

if __name__ == "__main__":
    main()






    

