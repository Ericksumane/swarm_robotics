"""Check one upstream PPO update using synthetic data, not robot rollouts."""

import numpy as np
import torch
from gymnasium.spaces import Box

from onpolicy.config import get_config
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy
from onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO


def snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def verify_update(module, before):
    parameters = list(module.parameters())
    assert all(torch.isfinite(p).all().item() for p in parameters)
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, parameters)
    ), "Parameters did not change."


def main():
    torch.manual_seed(42)
    rng = np.random.default_rng(42)

    args = get_config().parse_args([])
    args.algorithm_name = "mappo"
    args.use_recurrent_policy = False
    args.use_naive_recurrent_policy = False

    # Disable value normalization for this isolated update check.
    args.use_popart = False
    args.use_valuenorm = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 24

    local_space = Box(-np.inf, np.inf, shape=(12,), dtype=np.float32)
    global_space = Box(-np.inf, np.inf, shape=(36,), dtype=np.float32)
    action_space = Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    policy = R_MAPPOPolicy(
        args, local_space, global_space, action_space, device=device
    )
    trainer = R_MAPPO(args, policy, device=device)

    local_obs = rng.normal(size=(batch_size, 12)).astype(np.float32)
    global_obs = rng.normal(size=(batch_size, 36)).astype(np.float32)
    hidden = np.zeros(
        (batch_size, args.recurrent_N, args.hidden_size), dtype=np.float32
    )
    masks = np.ones((batch_size, 1), dtype=np.float32)

    with torch.no_grad():
        values, actions, log_probs, _, _ = policy.get_actions(
            global_obs, local_obs, hidden.copy(), hidden.copy(), masks
        )

    # Artificial targets test optimization only; they are not GAE returns.
    advantages = rng.normal(size=(batch_size, 1)).astype(np.float32)
    advantages = (
        advantages - advantages.mean()
    ) / (advantages.std() + 1e-5)

    old_values = values.cpu().numpy()
    returns = old_values + advantages

    sample = (
        global_obs,
        local_obs,
        hidden.copy(),
        hidden.copy(),
        actions.cpu().numpy(),
        old_values,
        returns,
        masks,
        masks.copy(),
        log_probs.cpu().numpy(),
        advantages,
        None,
    )

    actor_before = snapshot(policy.actor)
    critic_before = snapshot(policy.critic)

    results = trainer.ppo_update(sample)

    for result in results:
        assert torch.isfinite(torch.as_tensor(result)).all().item()

    verify_update(policy.actor, actor_before)
    verify_update(policy.critic, critic_before)

    print("Device:", device)
    print("Finite update outputs: PASSED")
    print("Actor parameter update: PASSED")
    print("Critic parameter update: PASSED")
    print("This check does not validate rollouts, GAE, or task learning.")


if __name__ == "__main__":
    main()
