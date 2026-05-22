import torch
import torch.optim as optim

from model_ppo import End2RaceActorCritic


def test_actor_mean_matches_plain_actor():
    torch.manual_seed(0)
    ac = End2RaceActorCritic(hidden_scale=1).eval()

    lidar = torch.rand(2, 5, 360)
    speed = torch.rand(2, 5, 1)
    hidden = torch.zeros(1, 2, ac.actor.gru.hidden_size)

    with torch.no_grad():
        dist, value, h1 = ac(lidar, speed, hidden.clone())
        actor_out, h2 = ac.actor(lidar, speed, hidden.clone())

    assert dist.mean.shape == (2, 5, 2)
    assert value.shape == (2, 5)
    assert torch.allclose(dist.mean, actor_out, atol=0, rtol=0)
    assert torch.allclose(h1, h2, atol=0, rtol=0)


def test_gru_history_affects_final_action():
    torch.manual_seed(1)
    ac = End2RaceActorCritic(hidden_scale=1).eval()

    lidar_a = torch.zeros(1, 6, 360)
    speed_a = torch.zeros(1, 6, 1)

    lidar_b = lidar_a.clone()
    speed_b = speed_a.clone()
    lidar_b[:, :5, :] = torch.rand(1, 5, 360)
    speed_b[:, :5, :] = torch.rand(1, 5, 1)
    lidar_b[:, 5, :] = lidar_a[:, 5, :]
    speed_b[:, 5, :] = speed_a[:, 5, :]

    with torch.no_grad():
        mean_a = ac(lidar_a, speed_a)[0].mean[:, -1]
        mean_b = ac(lidar_b, speed_b)[0].mean[:, -1]

    assert (mean_a - mean_b).abs().max().item() > 1e-6


def test_lidar_and_speed_affect_output():
    torch.manual_seed(2)
    ac = End2RaceActorCritic(hidden_scale=1).eval()

    speed = torch.ones(1, 4, 1) * 5.0
    lidar_near = torch.ones(1, 4, 360) * 0.5
    lidar_far = torch.ones(1, 4, 360) * 20.0

    with torch.no_grad():
        out_near = ac(lidar_near, speed)[0].mean
        out_far = ac(lidar_far, speed)[0].mean

    assert (out_near - out_far).abs().max().item() > 1e-6

    lidar = torch.rand(1, 4, 360)
    speed_zero = torch.zeros(1, 4, 1)
    speed_high = torch.ones(1, 4, 1) * 8.0

    with torch.no_grad():
        out_zero = ac(lidar, speed_zero)[0].mean
        out_high = ac(lidar, speed_high)[0].mean

    assert (out_zero - out_high).abs().max().item() > 1e-6


def test_log_std_optimizer_and_gradient():
    ac = End2RaceActorCritic(hidden_scale=1)
    critic_params = list(ac.value_head.parameters())
    critic_ids = {id(p) for p in critic_params}
    actor_params = [p for p in ac.parameters() if id(p) not in critic_ids]

    optimizer = optim.Adam(
        [
            {"params": critic_params, "lr": 1e-4},
            {"params": actor_params, "lr": 1e-5},
        ]
    )

    opt_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    model_ids = {id(p) for p in ac.parameters()}

    assert id(ac.log_std) in opt_ids
    assert opt_ids == model_ids

    lidar = torch.rand(1, 1, 360)
    speed = torch.rand(1, 1, 1)
    dist, _, _ = ac(lidar, speed)
    action = dist.mean.detach() + torch.tensor([[[0.01, 0.1]]])

    loss = -dist.log_prob(action).sum(-1).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    assert ac.log_std.grad is not None
    assert torch.isfinite(ac.log_std.grad).all()
