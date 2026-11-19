"""Check masking, permutation invariance and gradient propagation."""

import torch

from swarm_core.learning.neighbor_encoder import NeighborEncoder


def main():
    torch.manual_seed(42)
    model = NeighborEncoder()

    features = torch.randn(1, 3, 7)
    mask = torch.tensor([[1.0, 0.0, 1.0]])

    output = model(features, mask)
    assert output.shape == (1, 34)

    permutation = [2, 0, 1]
    reordered = model(features[:, permutation], mask[:, permutation])
    torch.testing.assert_close(output, reordered)

    changed = features.clone()
    changed[:, 1] = 1000.0
    torch.testing.assert_close(output, model(changed, mask))

    missing = model(features, torch.zeros_like(mask))
    torch.testing.assert_close(missing, torch.zeros_like(missing))

    empty = model(torch.empty(1, 0, 7), torch.empty(1, 0))
    torch.testing.assert_close(empty, torch.zeros_like(empty))

    output.square().sum().backward()
    gradients = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum().item() > 0 for g in gradients)

    print("Permutation invariance: PASSED")
    print("Invalid neighbor exclusion: PASSED")
    print("Missing and empty neighbor handling: PASSED")
    print("Encoder gradient propagation: PASSED")


if __name__ == "__main__":
    main()