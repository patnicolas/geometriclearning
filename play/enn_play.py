
import torch
import torch.nn as nn
from e3nn import o3


class SO3EquivariantEncoder(nn.Module):
    def __init__(self, input_dim=1, latent_dim=16):
        super().__init__()

        # Define the 'Irreps' (Irreducible Representations)
        # 1x0e = 1 scalar (even parity)
        # 1x1o = 1 vector (odd parity)
        self.input_irreps = o3.Irreps(f"{input_dim}x0e")

        # The latent space: a mix of scalars and vectors
        # This preserves the geometric orientation of the world
        self.latent_irreps = o3.Irreps(f"{latent_dim}x0e + {latent_dim}x1o")

        # Spherical Harmonics basis for the geometry
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=2)

        # Tensor Product: The core equivariant 'convolution'
        # It combines features with geometry while respecting SO(3)
        self.tp = o3.FullyConnectedTensorProduct(
            self.input_irreps,
            self.sh_irreps,
            self.latent_irreps
        )

    def forward(self, features, pos):
        """
        features: [batch, nodes, input_dim] (e.g., node mass or color)
        pos: [batch, nodes, 3] (the 3D coordinates in space)
        """
        # 1. Compute Spherical Harmonics of the relative positions
        # This encodes the 'direction' of points in the manifold
        sh = o3.spherical_harmonics(
            self.sh_irreps, pos, normalize=True, normalization='component'
        )

        # 2. Perform the Tensor Product
        # This mixes features with spatial orientation equivariantly
        latent_vectors = self.tp(features, sh)

        # 3. Global Pooling (Invariant or Equivariant)
        # Mean pooling across nodes preserves SO(3) equivariance
        world_latent = torch.mean(latent_vectors, dim=1)

        return world_latent

# --- Usage Example ---
# A world model input: 10 points in 3D space with 1 feature each
encoder = SO3EquivariantEncoder(input_dim=1, latent_dim=16)
dummy_features = torch.randn(1, 10, 1)
dummy_pos = torch.randn(1, 10, 3)

latent = encoder(dummy_features, dummy_pos)
print(f"Latent shape: {latent.shape}") # [1, 64] (16 scalars + 16*3 vectors)