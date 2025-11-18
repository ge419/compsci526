## Instructions

1. Run controller

```
cd controller
npm install
node index.js
```

2. Run frontend

```
cd frontend
npm install
npm run dev
```

  1. RBM Architecture and Energy Function

  The Bernoulli RBM is a bipartite undirected graphical model with two layers:

  Layers:
  - Visible layer v ∈ {0,1}^n_visible (430 ingredients in your case)
  - Hidden layer h ∈ {0,1}^n_hidden (500 latent features)

  Parameters:
  - Weight matrix W ∈ ℝ^(n_visible × n_hidden) - connections between visible and hidden units
  - Visible bias b_v ∈ ℝ^n_visible - bias for each ingredient
  - Hidden bias b_h ∈ ℝ^n_hidden - bias for each hidden unit

  Energy Function:

  The RBM defines an energy function:

  E(v, h) = -v^T W h - b_v^T v - b_h^T h

  Lower energy = more probable configuration

  Joint Probability Distribution:

  P(v, h) = (1/Z) exp(-E(v, h))

  where Z is the partition function (normalization constant):

  Z = Σ_v Σ_h exp(-E(v, h))

  ---
  2. Conditional Probabilities (Key Insight!)

  Due to the bipartite structure (no visible-to-visible or hidden-to-hidden connections), the conditional probabilities factorize:

  P(h | v) - Hidden given Visible:

  P(h_j = 1 | v) = σ(Σ_i W_ij v_i + b_h_j)
                = σ(W^T v + b_h)_j

  where σ(x) = 1/(1 + e^(-x)) is the sigmoid function.

  Code implementation (model/model/rbm.py:68-71):
  def sample_hidden(self, v):
      h_probs = self.sigmoid(np.dot(v, self.W) + self.hbias)
      h_samples = (np.random.rand(*h_probs.shape) < h_probs).astype(np.float32)
      return h_probs, h_samples

  P(v | h) - Visible given Hidden:

  P(v_i = 1 | h) = σ(Σ_j W_ij h_j + b_v_i)
                = σ(W h + b_v)_i

  Code implementation (model/model/rbm.py:84-87):
  def sample_visible(self, h):
      v_probs = self.sigmoid(np.dot(h, self.W.T) + self.vbias)
      v_samples = (np.random.rand(*v_probs.shape) < v_probs).astype(np.float32)
      return v_probs, v_samples

  ---
  3. Gibbs Sampling (Alternating Between Layers)

  Since we can't directly sample from P(v), we use Block Gibbs Sampling:

  v^(0) → h^(0) → v^(1) → h^(1) → v^(2) → ... → v^(T)

  Each transition involves:

  Step 1: Sample hidden from current visible
  For each j: h_j^(t) ~ Bernoulli(σ(W^T v^(t) + b_h))

  Step 2: Sample visible from new hidden
  For each i: v_i^(t+1) ~ Bernoulli(σ(W h^(t) + b_v))

  Mathematical Details:

  1. Compute activation probabilities:
  p_j = P(h_j = 1 | v^(t)) = σ((W^T v^(t) + b_h)_j)
  2. Stochastic sampling:
  h_j^(t) = {1 if u_j < p_j, else 0}  where u_j ~ Uniform(0,1)
  3. Repeat for visible layer:
  q_i = P(v_i = 1 | h^(t)) = σ((W h^(t) + b_v)_i)
  v_i^(t+1) = {1 if r_i < q_i, else 0}  where r_i ~ Uniform(0,1)

  Code implementation (model/model/rbm.py:253-258):
  # Gibbs sampling
  for _ in range(n_gibbs):  # Typically 1000 iterations
      h_probs, h_samples = self.sample_hidden(v)
      v_probs, v = self.sample_visible(h_samples)

  ---
  4. User Preference Conditioning

  Your implementation adds personalization by modulating the visible bias:

  Original bias:
  b_v ∈ ℝ^430

  User preference vector:
  u ∈ ℝ^latent_dim  (typically 50)

  Preference projection matrix:
  W_pref ∈ ℝ^(430 × 50)

  Modified bias:
  b_v' = b_v + α * (W_pref · u)

  where α ∈ [0,1] controls preference strength.

  Mathematical Effect:

  This changes the conditional probability:
  P(v_i = 1 | h) = σ(W h + b_v + α * (W_pref u))_i

  Ingredients aligned with user preferences get higher probabilities!

  Code implementation (model/model/rbm.py:318-330):
  # Create preference projection if not exists
  if not hasattr(self, 'W_pref'):
      latent_dim = len(user_pref)
      self.W_pref = np.random.randn(self.n_visible, latent_dim) * 0.01

  # Compute preference-modulated bias
  pref_bias = np.dot(self.W_pref, user_pref)

  # Temporarily modify visible bias
  self.vbias = self.vbias + alpha * pref_bias

  ---
  5. Dietary Restrictions (Ingredient Masking)

  After each Gibbs step, forbidden ingredients are zeroed out:

  v^(t+1) = v^(t+1) ⊙ mask

  where mask_i = 0 if ingredient i is forbidden, else 1.

  Mathematical effect:

  This enforces hard constraints:
  P(v_i = 1 | ...) = 0  if mask_i = 0

  Code implementation (model/model/rbm.py:361-368):
  # Apply ingredient mask to forbid certain ingredients
  if ingredient_mask is not None:
      v = v * ingredient_mask  # Zero out forbidden ingredients

  # Pin seed ingredients (keep them active)
  if seed_mask is not None:
      v = v + seed_mask
      v = np.clip(v, 0, 1)

  ---
  6. Complete Generation Pipeline

  Putting it all together, here's what happens when you generate a recipe:

  Initialization (t=0):

  Option A: Random initialization
  v^(0)_i ~ Bernoulli(0.1)  for all i

  Option B: Seed ingredients
  v^(0)_i = 1  if ingredient i is in seeds, else 0

  Gibbs Iterations (t=1 to T=1000):

  For t = 1 to 1000:
      # Hidden layer
      h^(t) ~ P(h | v^(t-1))

      # Visible layer
      v^(t) ~ P(v | h^(t))

      # Apply constraints
      v^(t) = v^(t) ⊙ mask_dietary        # Zero forbidden
      v^(t) = v^(t) + mask_seed            # Pin seeds
      v^(t) = clip(v^(t), 0, 1)            # Ensure [0,1]

  Final Sample:

  v^(1000) ≈ sample from P_model(v | preferences, constraints)

  Extract Top Ingredients:

  # Get indices where v_i > 0.5
  top_k = argsort(v^(1000))[::-1][:10]
  ingredients = [ingredient_names[i] for i in top_k if v^(1000)[i] > 0.5]

  ---
  7. Why Does This Work?

  Theoretical Guarantee:

  As T → ∞, the Gibbs sampling chain converges to the stationary distribution:
  v^(T) ~ P(v) = Σ_h P(v, h)

  In practice: 1000 steps is enough for approximate convergence.

  What the model learned:

  During training (Contrastive Divergence), the RBM learned:
  - W: Which ingredients co-occur (e.g., "tomato" + "pasta" + "basil")
  - b_v: Base popularity of each ingredient
  - b_h: Latent food patterns (Italian, Asian, desserts, etc.)

  ---
  8. Numerical Example

  Let's trace through one Gibbs step with simplified numbers:

  Given:
  - v = [1, 0, 1, 0, ...] (tomato=1, garlic=0, onion=1, ...)
  - W[0,:] = [0.5, -0.2, 0.8, ...] (tomato's connections)
  - b_h = [0.1, -0.3, 0.2, ...]

  Step 1: Sample hidden
  z = v @ W + b_h
    = [1, 0, 1, 0, ...] @ W + [0.1, -0.3, 0.2, ...]
    = [0.5 + 0.8 + 0.1, ...]  # contribution from tomato + onion + bias
    = [1.4, ...]

  h_probs = sigmoid(z) = [0.802, ...]
  h_sample = [1, ...] if random() < 0.802

  Step 2: Sample visible
  z' = h @ W^T + b_v
     = [1, ...] @ W^T + b_v

  v_probs = sigmoid(z')
  v_sample = Bernoulli(v_probs)

  ---
  Summary

  The RBM generates samples through:

  1. Learned joint distribution P(v, h) via energy function
  2. Block Gibbs sampling alternating between layers
  3. Preference conditioning shifting probabilities toward user taste
  4. Hard constraints (dietary, seeds) via masking
  5. Convergence to approximate P(v | user, constraints) after 1000 steps

  The beauty is that all these complex interactions (co-occurrence patterns, user preferences, constraints) emerge from the simple iterative process of sampling hidden → sampling visible!

> So this is a Bernoulli Bernoulli RBM? 

⏺ Yes, exactly! This is a Bernoulli-Bernoulli RBM.

  The naming convention is [Visible distribution]-[Hidden distribution], so:

  Bernoulli-Bernoulli RBM

  Visible layer (ingredients):
  - Binary: v_i ∈ {0, 1} (ingredient present or absent)
  - Conditional distribution: P(v_i = 1 | h) = σ(W_i^T h + b_v_i)
  - Sampling: v_i ~ Bernoulli(σ(...))

  Hidden layer (latent features):
  - Binary: h_j ∈ {0, 1} (latent feature active or not)
  - Conditional distribution: P(h_j = 1 | v) = σ(W^T_j v + b_h_j)
  - Sampling: h_j ~ Bernoulli(σ(...))

  You can see this in the sampling code:

  # Hidden units - Bernoulli
  h_probs = self.sigmoid(np.dot(v, self.W) + self.hbias)
  h_samples = (np.random.rand(*h_probs.shape) < h_probs).astype(np.float32)

  # Visible units - Bernoulli  
  v_probs = self.sigmoid(np.dot(h, self.W.T) + self.vbias)
  v_samples = (np.random.rand(*v_probs.shape) < v_probs).astype(np.float32)

  Both use the same sampling mechanism: compute probability with sigmoid, then sample binary values.

  ---
  Other RBM Types (for comparison)

  Gaussian-Bernoulli RBM:
  - Visible: real-valued (e.g., pixel intensities 0-255)
  - Hidden: binary
  - Used for: continuous data like images
  - Energy: E(v, h) = (v - b_v)^T (v - b_v) / (2σ²) - v^T W h - b_h^T h

  Bernoulli-Gaussian RBM:
  - Visible: binary
  - Hidden: real-valued
  - Less common

  ---
  Why Bernoulli-Bernoulli for Your Task?

  This is the right choice because:

  1. Ingredients are naturally binary - either in the dish or not
  2. Latent features are interpretable as binary - e.g., "is Italian?" "is spicy?" "is dessert?"
  3. Efficient computation - binary operations are fast
  4. Works well for sparse data - most dishes use only a small subset of 430 ingredients

  The class name BernoulliRBM is a common shorthand that typically implies Bernoulli-Bernoulli when used for binary data!

