# ⚡ 60-Second Quickstart

Get started with industrial-grade meta-learning in under a minute.

## Install

```bash
pip install meta-learning
```

## 60-Second Tutorial

### 1. Basic Few-Shot Learning

```python
import meta_learning as ml
import torch
import torch.nn as nn

# Create a simple feature extractor
feature_extractor = nn.Sequential(
    nn.Conv2d(1, 64, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten()
)

# Create ProtoNet model
model = ml.ProtoHead(feature_extractor)

# Load dataset and sample episode
dataset = ml.get_dataset("omniglot", split="train")
episode = ml.make_episode(dataset, n_way=5, k_shot=1, n_query=15)
support_x, support_y = episode['support_x'], episode['support_y']
query_x, query_y = episode['query_x'], episode['query_y']

# Run few-shot learning
logits = model(support_x, support_y, query_x) 
accuracy = (logits.argmax(-1) == query_y).float().mean()

print(f"5-way 1-shot accuracy: {accuracy:.3f}")
```

### 2. Command-Line Interface

The `mlfew` CLI provides a complete few-shot learning workflow:

```bash
# Train a model
mlfew fit --dataset omniglot --algorithm protonet --n-way 5 --k-shot 1

# Evaluate a trained model
mlfew eval --model checkpoints/protonet_omniglot.pt --dataset omniglot

# Run benchmarks
mlfew benchmark --datasets omniglot,miniimagenet --algorithms protonet,maml
```

### 3. Advanced Usage

```python
import meta_learning as ml

# Configure advanced ProtoNet
config = ml.PrototypicalConfig(
    distance_metric="squared_euclidean",
    temperature=1.0,
    use_cosine_classifier=False
)

protonet = ml.PrototypicalNetworks(feature_extractor, config)

# Configure MAML with a complete classifier
class SimpleClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = feature_extractor
        self.classifier = torch.nn.Linear(64, 5)  # 5-way classification
    def forward(self, x):
        return self.classifier(self.backbone(x))

classifier = SimpleClassifier()
maml_config = ml.MAMLConfig(
    inner_lr=0.01,
    inner_steps=5,
    first_order=False  # Second-order gradients
)

maml = ml.MAMLLearner(classifier, maml_config)

# Test-time compute scaling (NEW!) - 2024 breakthrough
ttc_config = ml.TestTimeComputeConfig(
    max_compute_budget=10,
    confidence_threshold=0.8,
    early_stopping=True,
    compute_allocation_strategy="adaptive"
)

ttc_scaler = ml.TestTimeComputeScaler(protonet, ttc_config)

# Run with adaptive compute allocation
logits, compute_stats = ttc_scaler(support_x, support_y, query_x, return_compute_stats=True)
print(f"Compute steps used: {compute_stats['compute_steps_used']}/{ttc_config.max_compute_budget}")
print(f"Final confidence: {compute_stats['final_confidence']:.3f}")
```

## Key Concepts

### Episodes in Few-Shot Learning

Few-shot learning uses **episodes** instead of traditional batches:

```python
# Episode structure
support_set = {
    "airplane": [img1, img2, img3],  # k_shot examples
    "car": [img4, img5, img6],
    "bird": [img7, img8, img9],
    # ... n_way classes total
}

query_set = {
    "airplane": [img10, img11, img12, img13, img14],  # n_query examples
    "car": [img15, img16, img17, img18, img19],
    "bird": [img20, img21, img22, img23, img24],
    # Test on same n_way classes
}
```

### Dataset Splits

Standard few-shot learning uses **class-disjoint** splits:

- **Train**: 64 classes (meta-training)
- **Val**: 16 classes (hyperparameter tuning)  
- **Test**: 20 classes (final evaluation)

Classes in test are completely unseen during training!

## Next Steps

=== "📚 Learn More"

    - [Algorithms Overview](algorithms.md) - Understand the methods
    - [Mathematical Foundation](research/math.md) - Deep dive into theory
    - [API Reference](api/core.md) - Complete function documentation

=== "💻 Try Examples"

    - [Basic Examples](examples/basic.md) - Simple use cases
    - [Advanced Features](examples/advanced.md) - Complex scenarios
    - [Custom Algorithms](examples/custom.md) - Build your own

=== "🔬 Research"

    - [Paper Implementations](research/papers.md) - Research accuracy
    - [Benchmarks](research/benchmarks.md) - Performance comparisons  
    - [Contributing](contributing.md) - Help improve the toolkit

## Common Issues

!!! tip "Import Errors"
    If you see import errors, ensure you have the required dependencies:
    ```bash
    pip install torch torchvision numpy scipy scikit-learn
    ```

!!! tip "Dataset Downloads"
    Datasets are automatically downloaded to `./data/` on first use. 
    For large datasets like miniImageNet, this may take time.

!!! tip "CUDA Support" 
    The toolkit automatically uses GPU if available:
    ```python
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    support_x = support_x.to(device)
    ```

Ready for more advanced usage? Check out the [Examples](examples/basic.md) section!