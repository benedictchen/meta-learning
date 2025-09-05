#!/usr/bin/env python3
"""
Demonstration of Advanced Information Bottleneck Features
Shows the enhanced mathematical implementations
"""

import numpy as np
from information_bottleneck import InformationBottleneck
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

def demo_advanced_features():
    # # Removed print spam: "...
    print("=" * 60)
    
    # Generate dataset
    X, Y = make_classification(
        n_samples=200,
        n_features=8,
        n_informative=5,
        n_classes=3,
        random_state=42
    )
    
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(Y))} classes\n")
    
    # 1. Standard Information Bottleneck
    print("1️⃣ Standard Information Bottleneck")
    ib1 = InformationBottleneck(n_clusters=4, beta=0.5, max_iter=20)
    results1 = ib1.fit(X, Y, use_annealing=False)
    print(f"   Final I(X;Z) = {results1['final_compression']:.4f}")
    print(f"   Final I(Z;Y) = {results1['final_prediction']:.4f}")
    
    # 2. Information Bottleneck with Annealing
    print("\n2️⃣ Information Bottleneck with Deterministic Annealing")
    ib2 = InformationBottleneck(n_clusters=4, beta=2.0, max_iter=20)
    results2 = ib2.fit(X, Y, use_annealing=True, annealing_schedule='exponential')
    print(f"   Final I(X;Z) = {results2['final_compression']:.4f}")
    print(f"   Final I(Z;Y) = {results2['final_prediction']:.4f}")
    
    # 3. Information Curve Analysis  
    print("\n3️⃣ Information Bottleneck Curve")
    beta_values = [0.1, 1.0, 5.0]
    curve_data = ib1.get_information_curve(beta_values, X[:100], Y[:100])
    
    print("   β\t\tI(X;Z)\t\tI(Z;Y)")
    for i, beta in enumerate(curve_data['beta_values']):
        print(f"   {beta}\t\t{curve_data['compression'][i]:.4f}\t\t{curve_data['prediction'][i]:.4f}")
    
    # 4. Cluster Analysis
    print("\n4️⃣ Learned Representation Analysis")
    Z_representation = ib2.transform(X)
    predictions = ib2.predict(X)
    accuracy = np.mean(predictions == Y)
    
    print(f"   Representation shape: {Z_representation.shape}")
    print(f"   Classification accuracy: {accuracy:.3f}")
    print(f"   Compression achieved: {results1['final_compression']/results2['final_compression']:.2f}x")
    
    # 5. Advanced MI Estimation Demo
    print("\n5️⃣ Advanced Mutual Information Estimation")
    
    # Create test data with known MI
    np.random.seed(42)
    x1 = np.random.randn(100)
    y1 = x1 + 0.1 * np.random.randn(100)  # High correlation
    
    x2 = np.random.randn(100)  
    y2 = np.random.randn(100)  # No correlation
    
    mi_high = ib1._estimate_mutual_info_continuous(x1.reshape(-1, 1), y1.reshape(-1, 1))
    mi_low = ib1._estimate_mutual_info_continuous(x2.reshape(-1, 1), y2.reshape(-1, 1))
    
    print(f"   MI (correlated data): {mi_high:.4f} bits")
    print(f"   MI (uncorrelated data): {mi_low:.4f} bits")
    print(f"   Ratio: {mi_high/mi_low:.2f}x (should be >> 1)")
    
    # Removed print spam: f"\n...
    # Removed print spam: f"\n...
    print(f"   ✓ Kraskov-Grassberger-Stögbauer continuous MI estimator")
    print(f"   ✓ Deterministic annealing with temperature scheduling")
    print(f"   ✓ Adaptive MI estimation (discrete/continuous)")
    print(f"   ✓ Early stopping and advanced convergence monitoring")
    print(f"   ✓ Information curve generation and analysis")
    print(f"   ✓ Enhanced clustering analysis")
    print(f"   ✓ Neural network extension (PyTorch-based)")

if __name__ == "__main__":
    demo_advanced_features()