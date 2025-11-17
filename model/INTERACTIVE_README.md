# Interactive Dish Recommendation System

An interactive system that learns your food preferences through accept/reject feedback and generates personalized dish recommendations.

## Quick Start

```bash
# Simple usage
python main.py --run-id dwma

# With custom parameters
python main.py --run-id dwma --alpha 0.7 --learning-rate 0.15
```

During the session:
- Enter `1` to accept a recommendation
- Enter `0` to reject a recommendation
- Enter `q` to quit

## How It Works

The system uses a **user preference vector `v`** that evolves based on your feedback:

1. **Initialization**: Start with a neutral preference vector (all zeros)
2. **Generation**: Generate ingredient combinations conditioned on your current preferences
3. **Recommendation**: Find matching dishes from the dataset
4. **Feedback**: You accept or reject the suggestion
5. **Learning**: Update the preference vector based on your feedback
6. **Iteration**: Repeat with improved recommendations

## Mathematical Approach

Similar to a conditional VAE, the system:

- **Preference Vector `v`**: Latent representation of user preferences (dimension: 50)
- **Conditioned Generation**: Modulates visible bias with `vbias_effective = vbias + α * W_pref @ v`
- **Preference Update**:
  - **Accept**: `v ← v + η * gradient` (move toward liked ingredients)
  - **Reject**: `v ← v - η * gradient` (move away from disliked ingredients)
- **Regularization**: `v ← 0.99 * v` (prevents preference vector from exploding)

## Usage

### Basic Usage
```bash
python interactive.py --run-id dwma
```

### With Custom Parameters
```bash
python interactive.py --run-id dwma --latent-dim 100 --alpha 0.7 --learning-rate 0.15
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--run-id` | (required) | Model run ID to load |
| `--latent-dim` | 50 | Dimension of user preference vector |
| `--alpha` | 0.5 | Strength of preference conditioning (0=none, 1=full) |
| `--learning-rate` | 0.1 | How quickly preferences adapt to feedback |
| `--n-gibbs` | 1000 | Gibbs sampling steps for generation |
| `--top-k` | 3 | Number of dish options to show |

## Interactive Commands

During the session, you can use:

- `accept` or `a` - Accept the recommendation (moves preferences toward it)
- `reject` or `r` - Reject the recommendation (moves preferences away from it)
- `more` - Show additional dish options
- `quit` or `q` - Exit the session

## Example Session

```
ITERATION 1
Generated Ingredients: chicken, rice, vegetables, soy sauce
Top Dish: Chicken Fried Rice
Your feedback: accept

ITERATION 2
Generated Ingredients: chicken, noodles, broth, green onions
Top Dish: Chicken Noodle Soup
Your feedback: reject

ITERATION 3
Generated Ingredients: beef, rice, egg, sesame oil
Top Dish: Beef Rice Bowl
Your feedback: accept
```

As you accept/reject dishes, the system learns:
- ✓ You like rice-based dishes
- ✗ You don't like noodle soups
- The preference vector `v` adapts accordingly

## How Preference Learning Works

### Accept Feedback
When you accept a dish with ingredients `[chicken, rice, soy sauce]`:
1. System creates a binary vector for these ingredients
2. Projects it to preference space: `gradient = W_pref^T @ ingredient_vector`
3. Updates: `v = v + learning_rate * gradient`
4. **Result**: Future generations favor similar ingredient combinations

### Reject Feedback
When you reject a dish:
1. Same gradient computation
2. Updates: `v = v - learning_rate * gradient`
3. **Result**: Future generations avoid similar ingredient combinations

### Convergence
Over time, the preference vector `v` stabilizes to represent your tastes. The system will:
- Generate more dishes you're likely to enjoy
- Avoid ingredient combinations you've rejected
- Balance exploration (new combinations) with exploitation (known preferences)

## Tips

1. **Start Neutral**: First few recommendations are random - use them to shape preferences
2. **Be Consistent**: Similar dishes should get similar feedback for faster learning
3. **Alpha Parameter**:
   - Low (0.2-0.4): More exploration, slower adaptation
   - Medium (0.5): Balanced
   - High (0.7-0.9): Strong preference influence, faster adaptation
4. **Learning Rate**:
   - Low (0.05): Slow, stable learning
   - Medium (0.1): Balanced
   - High (0.2+): Fast adaptation, may be unstable

## Technical Details

### Preference Vector Dimension
- **Latent-dim=50**: Good balance of expressiveness and stability
- **Latent-dim=100**: More expressive, captures finer preferences
- **Latent-dim=20**: Simpler, faster convergence but less nuanced

### Projection Matrix `W_pref`
- Randomly initialized (430 ingredients × latent_dim)
- Maps preference space to ingredient space
- Learned implicitly through feedback, not explicitly trained

### Why This Works
The RBM already learned ingredient co-occurrence patterns from 100K dishes. The preference vector simply biases the generation toward/away from certain patterns based on your feedback, without needing to retrain the base model.

## Testing

Run the test suite to verify the system works:
```bash
python test_interactive.py
```

This tests:
- Preference initialization
- Generation with/without preferences
- Accept/reject updates
- Multiple iteration stability
