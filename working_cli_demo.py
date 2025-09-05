# NOTE: episodic setting: freeze BN running stats or use Instance/LayerNorm
#!/usr/bin/env python3
"""
Working CLI Demo with Real Dataset
==================================

This is a complete working demo that uses CIFAR-10 (real dataset) to demonstrate
few-shot learning capabilities with our meta-learning library.

This replaces synthetic data with actual image classification tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import defaultdict
import argparse


class SimpleConvNet(nn.Module):
    """Simple CNN for CIFAR-10 few-shot learning."""
    
    def __init__(self, input_channels=3, hidden_dim=64, output_dim=5):
        super().__init__()
        self.features = nn.Sequential(
            # First conv block - FIX: GroupNorm instead of BatchNorm for few-shot
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),  # Better for few-shot learning
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second conv block - FIX: GroupNorm instead of BatchNorm
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Third conv block - FIX: GroupNorm instead of BatchNorm  
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        features = self.features(x)
        return self.classifier(features)


def load_cifar10_few_shot_data(n_way=5, n_support=5, n_query=15, data_dir="./data"):
    """
    Load CIFAR-10 data and create a few-shot learning episode.
    
    Args:
        n_way: Number of classes per episode
        n_support: Number of support examples per class
        n_query: Number of query examples per class  
        data_dir: Directory to download/store CIFAR-10 data
        
    Returns:
        Tuple of (support_x, support_y, query_x, query_y)
    """
    print(f"📁 Loading CIFAR-10 dataset from {data_dir}...")
    
    # Define transform for CIFAR-10
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load CIFAR-10 dataset (automatically downloads if not present)
    dataset = datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # CIFAR-10 class names for reference
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Group samples by class
    class_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(dataset):
        class_to_indices[label].append(idx)
    
    # Randomly select n_way classes
    available_classes = list(class_to_indices.keys())
    selected_classes = np.random.choice(available_classes, n_way, replace=False)
    
    # Removed print spam: f"...
    for i, cls in enumerate(selected_classes):
        print(f"   {i}: {class_names[cls]}")
    
    support_x, support_y = [], []
    query_x, query_y = [], []
    
    for new_label, original_class in enumerate(selected_classes):
        # Get indices for this class
        class_indices = class_to_indices[original_class]
        
        # Randomly sample support + query examples
        total_needed = n_support + n_query
        if len(class_indices) < total_needed:
            # Sample with replacement if not enough examples
            selected_indices = np.random.choice(class_indices, total_needed, replace=True)
        else:
            selected_indices = np.random.choice(class_indices, total_needed, replace=False)
        
        # Split into support and query
        support_indices = selected_indices[:n_support]
        query_indices = selected_indices[n_support:]
        
        # Add support examples
        for idx in support_indices:
            image, _ = dataset[idx]
            support_x.append(image)
            support_y.append(new_label)  # Use remapped label
        
        # Add query examples  
        for idx in query_indices:
            image, _ = dataset[idx]
            query_x.append(image)
            query_y.append(new_label)  # Use remapped label
    
    # Convert to tensors
    support_x = torch.stack(support_x)
    support_y = torch.tensor(support_y, dtype=torch.long)
    query_x = torch.stack(query_x)
    query_y = torch.tensor(query_y, dtype=torch.long)
    
    # Removed print spam: f"...
    print(f"   Support set: {support_x.shape} images, {len(support_y)} labels")
    print(f"   Query set: {query_x.shape} images, {len(query_y)} labels")
    
    return support_x, support_y, query_x, query_y


def train_few_shot_model(support_x, support_y, query_x, query_y, n_way=5):
    """
    Train a simple few-shot learning model on the support set and evaluate on query set.
    
    Args:
        support_x: Support images [n_support, 3, 32, 32]
        support_y: Support labels [n_support]
        query_x: Query images [n_query, 3, 32, 32]  
        query_y: Query labels [n_query]
        n_way: Number of classes
        
    Returns:
        Dictionary with training results
    """
    print(f"🏋️ Training few-shot model ({n_way}-way classification)...")
    
    # Create model
    model = SimpleConvNet(input_channels=3, hidden_dim=64, output_dim=n_way)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    training_losses = []
    
    # Train for a few epochs on support set
    for epoch in range(50):  # Quick training
        optimizer.zero_grad()
        
        # Forward pass
        support_logits = model(support_x)
        loss = criterion(support_logits, support_y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        training_losses.append(loss.item())
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch:2d}: Loss = {loss.item():.4f}")
    
    # Evaluate on query set
    model.eval()
    with torch.enable_grad():  # fixed: no_grad removed in inner loop
        query_logits = model(query_x)
        query_loss = criterion(query_logits, query_y)
        
        # Compute accuracy
        predictions = query_logits.argmax(dim=1)
        accuracy = (predictions == query_y).float().mean().item()
        
        # Per-class accuracy
        per_class_acc = []
        for class_id in range(n_way):
            class_mask = query_y == class_id
            if class_mask.sum() > 0:
                class_acc = (predictions[class_mask] == query_y[class_mask]).float().mean().item()
                per_class_acc.append(class_acc)
            else:
                per_class_acc.append(0.0)
    
    # Removed print spam: f"...
    print(f"   Final training loss: {training_losses[-1]:.4f}")
    print(f"   Query set accuracy: {accuracy:.1%}")
    print(f"   Per-class accuracy: {[f'{acc:.1%}' for acc in per_class_acc]}")
    
    return {
        'model': model,
        'training_losses': training_losses,
        'query_accuracy': accuracy,
        'query_loss': query_loss.item(),
        'per_class_accuracy': per_class_acc,
        'predictions': predictions,
        'query_labels': query_y
    }


def run_working_demo(args):
    """Run the complete working demo with real CIFAR-10 data."""
    # # Removed print spam: "...")
    print("=" * 60)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load real CIFAR-10 data for few-shot learning
    support_x, support_y, query_x, query_y = load_cifar10_few_shot_data(
        n_way=args.n_way,
        n_support=args.n_support,
        n_query=args.n_query,
        data_dir=args.data_dir
    )
    
    print()
    
    # Train and evaluate few-shot model
    results = train_few_shot_model(
        support_x, support_y, query_x, query_y, 
        n_way=args.n_way
    )
    
    print()
    # Removed print spam: "...
    print("-" * 30)
    print(f"Dataset: REAL CIFAR-10 (not synthetic!)")
    print(f"Task: {args.n_way}-way, {args.n_support}-shot classification")
    print(f"Query accuracy: {results['query_accuracy']:.1%}")
    print(f"Query loss: {results['query_loss']:.4f}")
    
    # Show some example predictions
    print()
    # Removed print spam: "...:")
    predictions = results['predictions'][:10]
    actual = results['query_labels'][:10]
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    available_classes = [0, 1, 2, 3, 4]  # Since we only use 5 classes
    
    for i in range(min(10, len(predictions))):
        pred_class = predictions[i].item()
        actual_class = actual[i].item()
        status = "✅" if pred_class == actual_class else "❌"
        print(f"   Query {i:2d}: Predicted class {pred_class}, Actual class {actual_class} {status}")
    
    print()
    # Removed print spam: "...
    # Removed print spam: f"...
    print(f"   not just synthetic random noise.")
    
    return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Meta-Learning CLI Demo with Real CIFAR-10 Dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--n-way", type=int, default=5,
        help="Number of classes per episode"
    )
    parser.add_argument(
        "--n-support", type=int, default=5,
        help="Number of support examples per class"
    )
    parser.add_argument(
        "--n-query", type=int, default=15,
        help="Number of query examples per class"
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data",
        help="Directory to store CIFAR-10 dataset"
    )
    
    args = parser.parse_args()
    
    # Run the working demo
    results = run_working_demo(args)
    
    # Exit successfully
    return 0


if __name__ == "__main__":
    exit(main())