import os
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import AGIQADataset, get_clip_transforms
from models import MSTCLIPIQA
from utils import compute_metrics

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data setup
    transform = get_clip_transforms()
    dataset = AGIQADataset(csv_file=args.csv_file, img_dir=args.img_dir, transform=transform)
    
    # We need to evaluate on the test set (20% split) to match train.py
    # To ensure the exact same split, we use the same seed and generator
    generator = torch.Generator().manual_seed(args.seed)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=generator)
    
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model setup
    model = MSTCLIPIQA(variant=args.variant).to(device)
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
        
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_targets = []

    print("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            mos = batch['mos'].to(device)
            prompts = batch['prompt'] if args.variant == 'B' else None

            preds = model(images, prompts)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(mos.cpu().numpy().flatten())

    metrics = compute_metrics(all_preds, all_targets)
    print("\nEvaluation Results:")
    print(f"SRCC: {metrics['srcc']:.4f}")
    print(f"PLCC: {metrics['plcc']:.4f}")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MST-CLIPIQA Model")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to dataset CSV file")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--variant", type=str, choices=['A', 'B'], default='A', help="Model variant ('A' for Template, 'B' for Prompt)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for dataset split")
    
    args = parser.parse_args()
    evaluate(args)
