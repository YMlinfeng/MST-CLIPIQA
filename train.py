import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from data import AGIQADataset, get_clip_transforms
from models import MSTCLIPIQA
from utils import CompositeLoss, compute_metrics

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data setup
    transform = get_clip_transforms()
    dataset = AGIQADataset(csv_file=args.csv_file, img_dir=args.img_dir, transform=transform)
    
    # 80/20 split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    # Use fixed seed for reproducibility
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model setup
    model = MSTCLIPIQA(variant=args.variant).to(device)
    
    # Optimizer setup - only parameters with requires_grad=True
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    
    # Loss setup
    criterion = CompositeLoss(lambda_rank=1.0, margin=0.1)
    
    best_srcc = -1.0
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for batch in progress_bar:
            images = batch['image'].to(device)
            prompts = batch['prompt'] if args.variant == 'B' else None
            mos = batch['mos'].to(device).float()
            
            optimizer.zero_grad()
            
            preds = model(images, prompts).squeeze(-1)
            loss = criterion(preds, mos)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                images = batch['image'].to(device)
                prompts = batch['prompt'] if args.variant == 'B' else None
                mos = batch['mos'].to(device).float()
                
                preds = model(images, prompts).squeeze(-1)
                loss = criterion(preds, mos)
                val_loss += loss.item()
                
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(mos.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        metrics = compute_metrics(val_preds, val_targets)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - SRCC: {metrics['srcc']:.4f} - PLCC: {metrics['plcc']:.4f}")
        
        if metrics['srcc'] > best_srcc:
            best_srcc = metrics['srcc']
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, f"best_model_variant_{args.variant}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved new best model with SRCC: {best_srcc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MST-CLIPIQA")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to dataset CSV file")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--variant", type=str, choices=['A', 'B'], default='A', help="Model variant (A: Template, B: Prompt)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    train(args)
