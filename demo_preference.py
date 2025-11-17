"""
Quick demo of the preference learning system (non-interactive).

Shows how the preference vector adapts over simulated accept/reject cycles.
"""

import numpy as np
from model.rbm import BernoulliRBM

print("="*80)
print("PREFERENCE LEARNING DEMO")
print("="*80)

# Load model
print("\nLoading model...")
rbm = BernoulliRBM.load_model('saved/dwma/model.npz')
print(f"Model loaded: {rbm}")

# Initialize user preference
print("\nInitializing user preference vector...")
user_pref = rbm.init_user_preference(latent_dim=50)
print(f"Initial preference norm: {np.linalg.norm(user_pref):.6f}")

# Simulate 10 interactions
print("\n" + "="*80)
print("SIMULATING 10 INTERACTIONS")
print("="*80)

np.random.seed(42)
for i in range(10):
    print(f"\nIteration {i+1}:")

    # Generate with current preferences
    sample = rbm.generate_with_preference(user_pref, n_samples=1, n_gibbs=500, alpha=0.5)
    ingredients = rbm.get_top_ingredients(sample[0], top_k=5)
    ingredient_names = [ing for ing, _ in ingredients]

    print(f"  Generated: {', '.join(ingredient_names)}")

    # Simulate feedback (alternating accept/reject for demo)
    feedback = 'accept' if i % 3 != 2 else 'reject'
    print(f"  Feedback: {feedback}")

    # Update preferences
    user_pref = rbm.update_user_preference(user_pref, ingredient_names, feedback, learning_rate=0.1)
    print(f"  Preference norm: {np.linalg.norm(user_pref):.6f}")

print("\n" + "="*80)
print("DEMO COMPLETE")
print("="*80)
print("\nKey observations:")
print("- Preference vector evolves based on accept/reject feedback")
print("- Norm increases with accepts, decreases with rejects")
print("- Regularization (0.99 decay) prevents unbounded growth")
print("\nRun the interactive system to try it yourself:")
print("  python interactive.py --run-id dwma")
