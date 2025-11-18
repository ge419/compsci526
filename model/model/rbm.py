import numpy as np
import json
from typing import List, Tuple, Optional


class BernoulliRBM:
    """
    Bernoulli Restricted Boltzmann Machine for ingredient generation.

    The RBM learns a probability distribution over binary ingredient vectors,
    where each visible unit represents the presence (1) or absence (0) of an ingredient.

    Attributes:
        n_visible (int): Number of visible units (ingredients)
        n_hidden (int): Number of hidden units
        W (np.ndarray): Weight matrix (n_visible x n_hidden)
        vbias (np.ndarray): Visible bias vector (n_visible,)
        hbias (np.ndarray): Hidden bias vector (n_hidden,)
        ingredients (list): List of ingredient names corresponding to visible units
        ingredient_to_idx (dict): Mapping from ingredient name to index
    """

    def __init__(self, n_visible: int, n_hidden: int = 500,
                 learning_rate: float = 0.01, ingredients: Optional[List[str]] = None):
        """
        Initialize the Bernoulli RBM.

        Args:
            n_visible: Number of visible units (number of unique ingredients)
            n_hidden: Number of hidden units (latent features)
            learning_rate: Learning rate for gradient descent
            ingredients: List of ingredient names (optional, for interpretability)
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate

        # Initialize weights with small random values (Xavier/Glorot initialization)
        self.W = np.random.randn(n_visible, n_hidden) * 0.01

        # Initialize biases to zero
        self.vbias = np.zeros(n_visible)
        self.hbias = np.zeros(n_hidden)

        # Store ingredient names for interpretability
        self.ingredients = ingredients if ingredients is not None else None
        if self.ingredients:
            self.ingredient_to_idx = {ing: idx for idx, ing in enumerate(self.ingredients)}
        else:
            self.ingredient_to_idx = {}

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with numerical stability."""
        # Clip to avoid overflow
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def sample_hidden(self, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample hidden units given visible units.

        Args:
            v: Visible unit activations (batch_size, n_visible)

        Returns:
            h_probs: Probabilities of hidden units being active
            h_samples: Binary samples from h_probs
        """
        # h_probs = sigmoid(v @ W + hbias)
        h_probs = self.sigmoid(np.dot(v, self.W) + self.hbias)
        h_samples = (np.random.rand(*h_probs.shape) < h_probs).astype(np.float32)
        return h_probs, h_samples

    def sample_visible(self, h: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample visible units given hidden units.

        Args:
            h: Hidden unit activations (batch_size, n_hidden)

        Returns:
            v_probs: Probabilities of visible units being active
            v_samples: Binary samples from v_probs
        """
        # v_probs = sigmoid(h @ W.T + vbias)
        v_probs = self.sigmoid(np.dot(h, self.W.T) + self.vbias)
        v_samples = (np.random.rand(*v_probs.shape) < v_probs).astype(np.float32)
        return v_probs, v_samples

    def contrastive_divergence(self, v0: np.ndarray, k: int = 1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform k-step Contrastive Divergence.

        Args:
            v0: Initial visible units (batch_size, n_visible)
            k: Number of Gibbs sampling steps

        Returns:
            v0: Initial visible state
            vk: Reconstructed visible state after k steps
            h0_probs: Hidden probabilities from v0
        """
        # Positive phase: compute hidden probabilities from data
        h0_probs, h0_samples = self.sample_hidden(v0)

        # Negative phase: k steps of Gibbs sampling
        hk_samples = h0_samples
        for _ in range(k):
            vk_probs, vk_samples = self.sample_visible(hk_samples)
            hk_probs, hk_samples = self.sample_hidden(vk_samples)

        # For the final reconstruction, we use probabilities (not samples) for visible units
        vk_probs, _ = self.sample_visible(hk_samples)

        return v0, vk_probs, h0_probs

    def train_batch(self, batch: np.ndarray, k: int = 1) -> float:
        """
        Train on a single batch using Contrastive Divergence.

        Args:
            batch: Batch of visible units (batch_size, n_visible)
            k: Number of CD steps

        Returns:
            reconstruction_error: Mean squared error between input and reconstruction
        """
        batch_size = batch.shape[0]

        # Perform contrastive divergence
        v0, vk, h0_probs = self.contrastive_divergence(batch, k=k)

        # Compute gradients
        # Positive phase: outer product of v0 and h0_probs
        positive_grad = np.dot(v0.T, h0_probs) / batch_size

        # Negative phase: outer product of vk and h(vk)
        hk_probs, _ = self.sample_hidden(vk)
        negative_grad = np.dot(vk.T, hk_probs) / batch_size

        # Update weights and biases
        self.W += self.learning_rate * (positive_grad - negative_grad)
        self.vbias += self.learning_rate * np.mean(v0 - vk, axis=0)
        self.hbias += self.learning_rate * np.mean(h0_probs - hk_probs, axis=0)

        # Compute reconstruction error
        reconstruction_error = np.mean((v0 - vk) ** 2)

        return reconstruction_error

    def fit(self, X: np.ndarray, n_epochs: int = 10, batch_size: int = 32,
            k: int = 1, verbose: bool = True, validation_data: Optional[np.ndarray] = None):
        """
        Train the RBM on data.

        Args:
            X: Training data (n_samples, n_visible)
            n_epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
            k: Number of CD steps
            verbose: Whether to print training progress
            validation_data: Optional validation data for monitoring

        Returns:
            history: Dictionary containing training history
        """
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size

        history = {
            'train_loss': [],
            'val_loss': [] if validation_data is not None else None
        }

        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]

            epoch_loss = 0.0
            for i in range(n_batches):
                batch = X_shuffled[i * batch_size:(i + 1) * batch_size]
                loss = self.train_batch(batch, k=k)
                epoch_loss += loss

            avg_train_loss = epoch_loss / n_batches
            history['train_loss'].append(avg_train_loss)

            # Validation
            val_loss_str = ""
            if validation_data is not None:
                val_loss = self.reconstruction_error(validation_data)
                history['val_loss'].append(val_loss)
                val_loss_str = f" - val_loss: {val_loss:.4f}"

            if verbose:
                print(f"Epoch {epoch+1}/{n_epochs} - loss: {avg_train_loss:.4f}{val_loss_str}")

        return history

    def reconstruction_error(self, X: np.ndarray) -> float:
        """
        Compute reconstruction error on data.

        Args:
            X: Data (n_samples, n_visible)

        Returns:
            Mean squared reconstruction error
        """
        h_probs, _ = self.sample_hidden(X)
        v_probs, _ = self.sample_visible(h_probs)
        return np.mean((X - v_probs) ** 2)

    def transform(self, X: np.ndarray, sample: bool = False) -> np.ndarray:
        """
        Transform visible data to hidden representation.

        Args:
            X: Visible data (n_samples, n_visible)
            sample: Whether to sample or use probabilities

        Returns:
            Hidden representation (n_samples, n_hidden)
        """
        h_probs, h_samples = self.sample_hidden(X)
        return h_samples if sample else h_probs

    def generate(self, n_samples: int = 1, n_gibbs: int = 1000,
                 seed_ingredients: Optional[List[str]] = None) -> np.ndarray:
        """
        Generate new ingredient combinations.

        Args:
            n_samples: Number of samples to generate
            n_gibbs: Number of Gibbs sampling steps
            seed_ingredients: Optional list of ingredients to start with

        Returns:
            Generated samples (n_samples, n_visible)
        """
        # Initialize visible units
        if seed_ingredients is not None and self.ingredients is not None:
            # Start from seed ingredients
            v = np.zeros((n_samples, self.n_visible))
            for ing in seed_ingredients:
                if ing in self.ingredient_to_idx:
                    idx = self.ingredient_to_idx[ing]
                    v[:, idx] = 1.0
        else:
            # Random initialization
            v = (np.random.rand(n_samples, self.n_visible) < 0.1).astype(np.float32)

        # Gibbs sampling
        for _ in range(n_gibbs):
            h_probs, h_samples = self.sample_hidden(v)
            v_probs, v = self.sample_visible(h_samples)

        return v

    def get_top_ingredients(self, sample: np.ndarray, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Get top k ingredients from a generated sample.

        Args:
            sample: Binary vector (n_visible,) or probability vector
            top_k: Number of top ingredients to return

        Returns:
            List of (ingredient_name, probability/activation) tuples
        """
        if self.ingredients is None:
            raise ValueError("Ingredient names not provided during initialization")

        # Get top k indices
        top_indices = np.argsort(sample)[::-1][:top_k]

        # Filter to only active ingredients (value > threshold)
        active_ingredients = [(self.ingredients[i], float(sample[i]))
                             for i in top_indices if sample[i] > 0.5]

        return active_ingredients

    def init_user_preference(self, latent_dim: int = 50) -> np.ndarray:
        """
        Initialize a user preference vector.

        Args:
            latent_dim: Dimension of the user preference vector

        Returns:
            Initialized user preference vector (latent_dim,)
        """
        # Initialize to zeros (neutral preferences)
        return np.zeros(latent_dim)

    def generate_with_preference(self, user_pref: np.ndarray, n_samples: int = 1,
                                 n_gibbs: int = 1000, alpha: float = 0.5,
                                 ingredient_mask: Optional[np.ndarray] = None,
                                 seed_ingredients: Optional[List[str]] = None) -> np.ndarray:
        """
        Generate ingredient combinations conditioned on user preferences.

        The user preference vector modulates the visible bias to shift the
        generation distribution toward user preferences.

        Args:
            user_pref: User preference vector (latent_dim,)
            n_samples: Number of samples to generate
            n_gibbs: Number of Gibbs sampling steps
            alpha: Strength of preference conditioning (0 = no effect, 1 = full effect)
            ingredient_mask: Optional binary mask (n_visible,) where 0 = forbidden, 1 = allowed
            seed_ingredients: Optional list of ingredient names to start generation with

        Returns:
            Generated samples (n_samples, n_visible)
        """
        # Create preference projection matrix if not exists
        if not hasattr(self, 'W_pref'):
            latent_dim = len(user_pref)
            # Random projection from preference space to visible space
            self.W_pref = np.random.randn(self.n_visible, latent_dim) * 0.01

        # Compute preference-modulated bias
        pref_bias = np.dot(self.W_pref, user_pref)

        # Save original bias
        original_vbias = self.vbias.copy()

        # Temporarily modify visible bias with user preferences
        self.vbias = self.vbias + alpha * pref_bias

        # Initialize visible units with seed ingredients if provided
        seed_mask = None
        if seed_ingredients is not None and self.ingredients is not None:
            v = np.zeros((n_samples, self.n_visible), dtype=np.float32)
            seed_mask = np.zeros(self.n_visible, dtype=np.float32)

            for ing in seed_ingredients:
                ing_lower = ing.lower()
                # Try exact match first
                if ing_lower in self.ingredient_to_idx:
                    idx = self.ingredient_to_idx[ing_lower]
                    v[:, idx] = 1.0
                    seed_mask[idx] = 1.0  # Mark as seed ingredient
                else:
                    # Try case-insensitive match
                    for model_ing, idx in self.ingredient_to_idx.items():
                        if model_ing.lower() == ing_lower:
                            v[:, idx] = 1.0
                            seed_mask[idx] = 1.0  # Mark as seed ingredient
                            break
        else:
            # Random initialization
            v = (np.random.rand(n_samples, self.n_visible) < 0.1).astype(np.float32)

        # Gibbs sampling
        for _ in range(n_gibbs):
            h_probs, h_samples = self.sample_hidden(v)
            v_probs, v = self.sample_visible(h_samples)

            # Apply ingredient mask to forbid certain ingredients
            if ingredient_mask is not None:
                v = v * ingredient_mask  # Zero out forbidden ingredients

            # Pin seed ingredients (keep them active)
            if seed_mask is not None:
                v = v + seed_mask  # Add seed ingredients back
                v = np.clip(v, 0, 1)  # Ensure values stay in [0, 1]

        # Restore original bias
        self.vbias = original_vbias

        return v

    def update_user_preference(self, user_pref: np.ndarray, ingredients: List[str],
                               feedback: str, learning_rate: float = 0.1) -> np.ndarray:
        """
        Update user preference vector based on accept/reject feedback.

        Args:
            user_pref: Current user preference vector (latent_dim,)
            ingredients: List of ingredient names in the shown sample
            feedback: 'accept' or 'reject'
            learning_rate: Learning rate for preference update

        Returns:
            Updated user preference vector
        """
        # Create ingredient vector
        ingredient_vec = np.zeros(self.n_visible)
        for ing in ingredients:
            if ing in self.ingredient_to_idx:
                idx = self.ingredient_to_idx[ing]
                ingredient_vec[idx] = 1.0

        # Create preference projection matrix if not exists
        if not hasattr(self, 'W_pref'):
            latent_dim = len(user_pref)
            self.W_pref = np.random.randn(self.n_visible, latent_dim) * 0.01

        # Compute gradient: how should we update preferences to increase/decrease
        # probability of this ingredient combination?

        # Project ingredient vector to preference space
        gradient = np.dot(self.W_pref.T, ingredient_vec - 0.5)  # Center around 0.5

        # Update based on feedback
        if feedback == 'accept':
            # Move preferences toward this combination
            user_pref = user_pref + learning_rate * gradient
        elif feedback == 'reject':
            # Move preferences away from this combination
            user_pref = user_pref - learning_rate * gradient

        # Optional: L2 regularization to prevent preference vector from growing too large
        user_pref = user_pref * 0.99

        return user_pref

    def save(self, filepath: str):
        """
        Save the RBM model to a file.

        Args:
            filepath: Path to save the model (will be saved as .npz)
        """
        np.savez(filepath,
                 W=self.W,
                 vbias=self.vbias,
                 hbias=self.hbias,
                 n_visible=self.n_visible,
                 n_hidden=self.n_hidden,
                 learning_rate=self.learning_rate,
                 ingredients=self.ingredients if self.ingredients else [])
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load the RBM model from a file.

        Args:
            filepath: Path to the saved model
        """
        data = np.load(filepath, allow_pickle=True)
        self.W = data['W']
        self.vbias = data['vbias']
        self.hbias = data['hbias']
        self.n_visible = int(data['n_visible'])
        self.n_hidden = int(data['n_hidden'])
        self.learning_rate = float(data['learning_rate'])

        ingredients_array = data['ingredients']
        if len(ingredients_array) > 0:
            self.ingredients = list(ingredients_array)
            self.ingredient_to_idx = {ing: idx for idx, ing in enumerate(self.ingredients)}

        import sys
        print(f"Model loaded from {filepath}", file=sys.stderr)
        return self

    @classmethod
    def load_model(cls, filepath: str):
        """
        Load a saved RBM model from file (class method).

        Args:
            filepath: Path to the saved model (.npz file)

        Returns:
            BernoulliRBM: A new RBM instance with loaded weights

        Example:
            rbm = BernoulliRBM.load_model('saved/rbm_h500_e20.npz')
        """
        # Load data from file
        data = np.load(filepath, allow_pickle=True)

        n_visible = int(data['n_visible'])
        n_hidden = int(data['n_hidden'])
        learning_rate = float(data['learning_rate'])

        ingredients_array = data['ingredients']
        ingredients = list(ingredients_array) if len(ingredients_array) > 0 else None

        # Create new instance
        rbm = cls(n_visible=n_visible,
                  n_hidden=n_hidden,
                  learning_rate=learning_rate,
                  ingredients=ingredients)

        # Load weights and biases
        rbm.W = data['W']
        rbm.vbias = data['vbias']
        rbm.hbias = data['hbias']

        import sys
        print(f"Model loaded from {filepath}", file=sys.stderr)
        print(f"  n_visible={n_visible}, n_hidden={n_hidden}", file=sys.stderr)
        if ingredients:
            print(f"  {len(ingredients)} ingredient names loaded", file=sys.stderr)

        return rbm

    def __repr__(self):
        return f"BernoulliRBM(n_visible={self.n_visible}, n_hidden={self.n_hidden}, lr={self.learning_rate})"
