import os
import argparse
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
from model import End2Race, End2RaceDualHead, End2RaceDeep, End2RaceDeepDualHead


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train End2Race speed-conditioned model')

    # Data — default path is the original single-dir loader; --loading_type overrides it with a manifest.
    parser.add_argument("--data_path", type=str, default="Dataset_Austin/success",
                        help="Single success/ directory of CSVs (used when --loading_type is not given).")
    parser.add_argument("--dataset_dir", type=str, default="Dataset_Austin",
                        help="Dataset root that contains manifest_<mode>.csv (only used with --loading_type).")
    parser.add_argument("--loading_type", type=str, default=None,
                        choices=('best_group', 'best_200', 'follow_first', 'overtake_first', 'merge'),
                        help="If set, load manifest_<type>.csv from --dataset_dir instead of globbing --data_path.")

    # Model
    parser.add_argument("--model_path", type=str, default="end2race.pth")
    parser.add_argument("--model_type", type=str, default='base', choices=('base', 'dual_head', 'deep', 'deep_dual'), help="Model architecture variant.")
    parser.add_argument("--hidden_scale", type=int, default=4)
    parser.add_argument("--mask_prob", type=float, default=0.1)

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=300)

    return parser.parse_args()


def read_manifest(dataset_dir: str, mode: str) -> List[str]:
    """Read <dataset_dir>/manifest_<mode>.csv and return absolute CSV paths."""
    manifest_path = os.path.join(dataset_dir, f'manifest_{mode}.csv')
    df = pd.read_csv(manifest_path)
    return [os.path.join(dataset_dir, p) for p in df['path'].tolist()]


class SequenceDataset(Dataset):

    def __init__(self, csv_paths: List[str], stride: int = 1, device: str = "cuda"):
        self.stride = stride
        self.device = device

        # Updated column names to match new CSV format
        self.lidar_columns = [f"lidar_{i}" for i in range(360)]
        self.action_columns = ["steer", "desired_speed"]

        self.sequence_length = self._determine_sequence_length(csv_paths[0])
        self.sequences = []
        self._load_episodes(csv_paths)

        print(f"Loaded {len(self.sequences)} sequences from {len(csv_paths)} CSVs")

    def _load_episodes(self, csv_paths: List[str]):
        """Load CSV files and create sequences."""
        for csv_file in csv_paths:
            df = pd.read_csv(csv_file)

            # Updated to match new CSV format: time, steer, desired_speed, lidar_0, ..., lidar_359
            required_cols = self.lidar_columns + self.action_columns
            if not all(col in df.columns for col in required_cols):
                continue

            # Minimum length needs one extra for speed alignment
            min_length = self.sequence_length + 1
            if len(df) < min_length:
                continue

            lidar_data = df[self.lidar_columns].values.astype(np.float32)
            action_data = df[self.action_columns].values.astype(np.float32)

            self._create_sequences(lidar_data, action_data)

    def _determine_sequence_length(self, first_csv: str) -> int:
        """Automatically determine sequence length from the first CSV file in the dataset"""
        df = pd.read_csv(first_csv)
        sequence_length = len(df) - 1
        print(f"Sequence_length: {sequence_length}")
        return sequence_length

    def _create_sequences(self, lidar_data: np.ndarray, action_data: np.ndarray):
        """Create sequences with speed conditioning."""
        # Skip first timestep for speed alignment
        lidar_valid = lidar_data[1:]
        action_valid = action_data[1:]
        # Use previous speed as input (desired_speed is column 1)
        speed_prev = action_data[:-1, 1:2]  # Take desired_speed column and keep 2D shape
        num_samples = len(lidar_valid)

        for end_idx in range(self.sequence_length - 1, num_samples, self.stride):
            start_idx = end_idx - self.sequence_length + 1
            self.sequences.append({
                'lidar': lidar_valid[start_idx:end_idx + 1],
                'speed': speed_prev[start_idx:end_idx + 1],
                'action': action_valid[start_idx:end_idx + 1]
            })

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence = self.sequences[idx]

        lidar_tensor = torch.tensor(sequence['lidar'], dtype=torch.float32, device=self.device)
        speed_tensor = torch.tensor(sequence['speed'], dtype=torch.float32, device=self.device)
        action_tensor = torch.tensor(sequence['action'], dtype=torch.float32, device=self.device)

        return lidar_tensor, speed_tensor, action_tensor


def train(model_path, model, train_loader, criterion, optimizer, scheduler, num_epochs=100):
    """Unified training function."""
    best_loss = float("inf")
    epoch_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for lidar_seq, speed_seq, target_actions in pbar:
                optimizer.zero_grad()

                predicted_actions, _ = model(lidar_seq, speed_seq)

                # Flatten for loss calculation
                predicted_actions_flat = predicted_actions.view(-1, predicted_actions.shape[-1])
                target_actions_flat = target_actions.view(-1, target_actions.shape[-1])

                steer_loss = criterion(predicted_actions_flat[:, 0], target_actions_flat[:, 0])
                speed_loss = criterion(predicted_actions_flat[:, 1], target_actions_flat[:, 1])
                loss = steer_loss + speed_loss * 0.05

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.5f}")

        scheduler.step(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            print(f"  New best loss: {best_loss:.5f}. Model saved.")

if __name__ == "__main__":
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.loading_type:
        csv_paths = read_manifest(args.dataset_dir, args.loading_type)
        print(f"Manifest {args.loading_type}: {len(csv_paths)} CSVs")
    else:
        csv_paths = sorted(glob.glob(os.path.join(args.data_path, "*.csv")))
        print(f"Data path {args.data_path}: {len(csv_paths)} CSVs")

    dataset = SequenceDataset(
        csv_paths=csv_paths,
        stride=1,
        device=device
    )

    dataloader_kwargs = {'pin_memory': False, 'num_workers': 0} if device.type != "cpu" else {}
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **dataloader_kwargs
    )

    # Create speed-conditioned model
    model_cls = {
        'base':      End2Race,
        'dual_head': End2RaceDualHead,
        'deep':      End2RaceDeep,
        'deep_dual': End2RaceDeepDualHead,
    }[args.model_type]
    model = model_cls(mask_prob=args.mask_prob, hidden_scale=args.hidden_scale).to(device)
    print(f"Model: {model_cls.__name__}")

    print(f"Train batches: {len(train_loader)}")

    # Load pretrained weights if available
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=False))

    # Setup optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    # Train model
    train(
        model_path=args.model_path,
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=args.num_epochs
    )

    print("\nTraining completed successfully!")
