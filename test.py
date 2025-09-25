import jax.numpy as jnp
import jax.random as random

# Example to understand the shapes
def sample_K_with_explanation(rng, mu, std, K):
    print(f"Input shapes:")
    print(f"  mu.shape = {mu.shape}")
    print(f"  std.shape = {std.shape}")
    print(f"  K = {K}")
    print()
    
    # Step 1: Create sample_shape
    sample_shape = (K,) + mu.shape
    print(f"Step 1 - Building sample_shape:")
    print(f"  (K,) = {(K,)}")
    print(f"  mu.shape = {mu.shape}")
    print(f"  sample_shape = (K,) + mu.shape = {sample_shape}")
    print()
    
    # Step 2: Sample noise
    eps = random.normal(rng, sample_shape)
    print(f"Step 2 - Sample noise:")
    print(f"  eps = random.normal(rng, {sample_shape})")
    print(f"  eps.shape = {eps.shape}")
    print()
    
    # Step 3: Reparameterization
    print(f"Step 3 - Reparameterization with broadcasting:")
    print(f"  z = mu + eps * std")
    print(f"  mu.shape = {mu.shape}")
    print(f"  eps.shape = {eps.shape}")
    print(f"  std.shape = {std.shape}")
    
    z = mu + eps * std
    print(f"  z.shape = {z.shape}")
    print()
    
    return z

# Example usage with different scenarios
print("="*60)
print("EXAMPLE 1: 1D latent space")
print("="*60)

# Case 1: Simple 1D latent variable
rng = random.PRNGKey(42)
mu_1d = jnp.array([2.0, -1.0, 0.5])  # 3-dimensional latent space
std_1d = jnp.array([1.0, 0.5, 2.0])
K = 5

z1 = sample_K_with_explanation(rng, mu_1d, std_1d, K)
print(f"Final result z:")
print(f"  Shape: {z1.shape}")
print(f"  Interpretation: {K} samples of {mu_1d.shape[0]}-dimensional latent vectors")
print()

print("="*60)
print("EXAMPLE 2: 2D latent space (like image features)")
print("="*60)

# Case 2: 2D latent variable (like image features)
mu_2d = jnp.ones((4, 6))  # 4x6 latent feature map
std_2d = jnp.ones((4, 6)) * 0.5
K = 3

rng, subkey = random.split(rng)
z2 = sample_K_with_explanation(subkey, mu_2d, std_2d, K)
print(f"Final result z:")
print(f"  Shape: {z2.shape}")
print(f"  Interpretation: {K} samples of {mu_2d.shape} latent feature maps")
print()

print("="*60)
print("VISUAL REPRESENTATION")
print("="*60)
print("For the 1D case:")
print("mu    shape: (3,)     ->  [μ₁, μ₂, μ₃]")
print("eps   shape: (5, 3)   ->  [[ε₁₁, ε₁₂, ε₁₃],")
print("                          [ε₂₁, ε₂₂, ε₂₃],")
print("                          [ε₃₁, ε₃₂, ε₃₃],")
print("                          [ε₄₁, ε₄₂, ε₄₃],")
print("                          [ε₅₁, ε₅₂, ε₅₃]]")
print()
print("z     shape: (5, 3)   ->  [[μ₁+ε₁₁σ₁, μ₂+ε₁₂σ₂, μ₃+ε₁₃σ₃],  <- sample 1")
print("                          [μ₁+ε₂₁σ₁, μ₂+ε₂₂σ₂, μ₃+ε₂₃σ₃],  <- sample 2")
print("                          [μ₁+ε₃₁σ₁, μ₂+ε₃₂σ₂, μ₃+ε₃₃σ₃],  <- sample 3")
print("                          [μ₁+ε₄₁σ₁, μ₂+ε₄₂σ₂, μ₃+ε₄₃σ₃],  <- sample 4")
print("                          [μ₁+ε₅₁σ₁, μ₂+ε₅₂σ₂, μ₃+ε₅₃σ₃]]  <- sample 5")