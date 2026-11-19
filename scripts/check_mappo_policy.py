"""Check upstream MAPPO policy construction and inference on synthetic inputs."""

import numpy as np
import torch
from gymnasium.spaces import Box

from onpolicy.config import get_config
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy


def main():
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    args = get_config().parse_args([])
    args.algorithm_name = "mappo"
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Synthetic dimensions used only for this integration check.
    num_agents = 3
    local_dim = 12
    global_dim = num_agents * local_dim

    local_space = Box(-np.inf, np.inf, shape=(local_dim,), dtype=np.float32)
    global_space = Box(-np.inf, np.inf, shape=(global_dim,), dtype=np.float32)
    action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    policy = R_MAPPOPolicy(
        args, local_space, global_space, action_space, device=device
    )

    local_obs = rng.normal(size=(num_agents, local_dim)).astype(np.float32)
    global_obs = np.tile(local_obs.reshape(1, -1), (num_agents, 1))

    hidden = np.zeros(
        (num_agents, args.recurrent_N, args.hidden_size), dtype=np.float32
    )
    masks = np.ones((num_agents, 1), dtype=np.float32)

    policy.actor.eval()
    policy.critic.eval()

    with torch.no_grad():
        values, actions, log_probs, _, _ = policy.get_actions(
            global_obs,
            local_obs,
            hidden.copy(),
            hidden.copy(),
            masks,
            deterministic=True,
        )

        # Deployment inference requires local inputs only.
        local_actions, _ = policy.act(
            local_obs, hidden.copy(), masks, deterministic=True
        )

    assert actions.shape == (num_agents, 2)
    assert values.shape == (num_agents, 1)
    for tensor in (values, actions, log_probs):
        assert torch.isfinite(tensor).all().item()
    assert torch.allclose(actions, local_actions)

    print("Device:", device)
    print("Actions shape:", tuple(actions.shape))
    print("Values shape:", tuple(values.shape))
    print("Local-only actor inference: PASSED")
    print("MAPPO policy construction and inference: PASSED")
    print("This check does not validate training or robot control.")


if __name__ == "__main__":
    main()
