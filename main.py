import argparse
from train import train
from eval import evaluate

def main():
    parser = argparse.ArgumentParser(description="MST-CLIPIQA: Multi-Scale Two-Stream Vision-Language Alignment for AIGIQA")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train parser
    train_parser = subparsers.add_parser("train", help="Train the MST-CLIPIQA model")
    train_parser.add_argument("--csv_file", type=str, required=True, help="Path to dataset CSV file")
    train_parser.add_argument("--img_dir", type=str, required=True, help="Path to image directory")
    train_parser.add_argument("--variant", type=str, choices=['A', 'B'], default='A', help="Model variant: 'A' (Template) or 'B' (Prompt)")
    train_parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    train_parser.add_argument("--weight_decay", type=float, default=1e-3, help="Weight decay")
    train_parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    train_parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    train_parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    
    # Eval parser
    eval_parser = subparsers.add_parser("eval", help="Evaluate the MST-CLIPIQA model")
    eval_parser.add_argument("--csv_file", type=str, required=True, help="Path to dataset CSV file")
    eval_parser.add_argument("--img_dir", type=str, required=True, help="Path to image directory")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    eval_parser.add_argument("--variant", type=str, choices=['A', 'B'], default='A', help="Model variant: 'A' (Template) or 'B' (Prompt)")
    eval_parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    eval_parser.add_argument("--num_workers", type=int, default=4, help="Number of dataloader workers")
    eval_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args)
    elif args.command == "eval":
        evaluate(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
