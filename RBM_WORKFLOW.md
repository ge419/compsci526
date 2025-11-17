# RBM Training and Generation Workflow

## Overview
Each training run generates a unique 4-letter run ID (e.g., `dwma`, `abcd`) and saves all artifacts in a dedicated directory under `saved/{run_id}/`.

## Directory Structure
```
saved/
├── dwma/
│   ├── model.npz              # Trained RBM weights
│   ├── training_history.png   # Loss curves
│   └── config.json            # Training configuration and metrics
├── abcd/
│   ├── model.npz
│   ├── training_history.png
│   └── config.json
└── ...
```

## Training a Model

### Basic Training
```bash
python train.py --n-epochs 20 --n-hidden 500
```

### Custom Training
```bash
python train.py \
    --n-epochs 50 \
    --n-hidden 800 \
    --learning-rate 0.005 \
    --batch-size 128 \
    --min-count 100 \
    --cd-steps 5 \
    --test-generation
```

### Key Parameters
- `--n-hidden`: Number of hidden units (default: 500)
- `--n-epochs`: Training epochs (default: 20)
- `--learning-rate`: Learning rate (default: 0.01)
- `--batch-size`: Batch size (default: 64)
- `--min-count`: Minimum ingredient occurrence count (default: 50)
- `--cd-steps`: Contrastive Divergence steps (default: 1)
- `--test-generation`: Run generation tests after training

### Output
After training, you'll see:
```
Run ID: dwma
Run directory: saved/dwma/

Saved files:
  - Model: saved/dwma/model.npz
  - Training plot: saved/dwma/training_history.png
  - Configuration: saved/dwma/config.json
```

## Generating Ingredients

### Basic Generation
```bash
python generate.py --run-id dwma --n-samples 5
```

### Generation with Dish Matching
```bash
python generate.py --run-id dwma --n-samples 3 --show-dishes --top-k 3
```

### Reproducible Generation
```bash
python generate.py --run-id dwma --n-samples 5 --seed 42
```

### Key Parameters
- `--run-id`: Run ID to load (required, e.g., `dwma`)
- `--n-samples`: Number of ingredient combinations to generate (default: 5)
- `--n-gibbs`: Gibbs sampling steps (default: 1000)
- `--seed`: Random seed for reproducibility (default: None, random each time)
- `--show-dishes`: Show matching dishes from dataset
- `--top-k`: Number of top matching dishes to show (default: 3)

## Using Models in Python

### Load a Model
```python
from model.rbm import BernoulliRBM

# Load by run ID
rbm = BernoulliRBM.load_model('saved/dwma/model.npz')
```

### Generate Ingredients
```python
# Generate 5 random combinations
samples = rbm.generate(n_samples=5, n_gibbs=1000)

# Get top ingredients for first sample
ingredients = rbm.get_top_ingredients(samples[0], top_k=10)
for ing, prob in ingredients:
    print(f"  - {ing} ({prob:.2f})")
```

### Load Configuration
```python
import json

with open('saved/dwma/config.json') as f:
    config = json.load(f)

print(f"Test loss: {config['test_loss']}")
print(f"Hidden units: {config['n_hidden']}")
```

## Tips

1. **Experiment Tracking**: Each run gets a unique ID, so you can easily track and compare experiments
2. **Configuration File**: The `config.json` stores all hyperparameters and metrics for reproducibility
3. **Training Plot**: Visualize learning progress in `training_history.png`
4. **Gibbs Steps**: More steps (e.g., 1000-2000) generally produce better samples but take longer
5. **Dish Matching**: Use `--show-dishes` to find real dishes from the dataset that match generated ingredients
