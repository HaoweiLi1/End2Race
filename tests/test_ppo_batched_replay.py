"""Unit tests for the batched PPO training actor replay (Phase 5B).

The reference inside each test is an independent per-slot batch-size-one
loop, so index mapping, hidden alignment, padding behaviour, and output
restoration are checked against a second implementation rather than against
the code under test itself.
"""

import unittest

import torch

from model import End2Race
from ppo.policy import (
    END2RACE_LIDAR_SIZE,
    END2RACE_OBSERVATION_SIZE,
    End2RaceGRUPolicy,
)


class _ActorStub:
    """Minimal duck-typed host for the policy's actor-forward methods."""

    def __init__(self, actor: End2Race) -> None:
        self.end2race_actor = actor

    @property
    def actor_hidden_size(self) -> int:
        return self.end2race_actor.gru.hidden_size

    @staticmethod
    def _actor_observation(obs: torch.Tensor) -> torch.Tensor:
        return obs


def _make_stub(seed: int = 0) -> _ActorStub:
    torch.manual_seed(seed)
    actor = End2Race(mask_prob=0.0, hidden_scale=1)
    actor.eval()
    return _ActorStub(actor)


def _make_inputs(n_seq: int, max_length: int, seed: int = 1):
    generator = torch.Generator().manual_seed(seed)
    obs = torch.rand(n_seq * max_length, END2RACE_OBSERVATION_SIZE, generator=generator) * 5.0
    hidden = torch.randn(1, n_seq, 420, generator=generator) * 0.1
    episode_starts = torch.zeros(n_seq * max_length)
    return obs, hidden, episode_starts


def _validity(lengths: list[int], max_length: int):
    return tuple(
        tuple(step < length for length in lengths) for step in range(max_length)
    )


def _reference_replay(stub, obs, hidden, episode_starts, lengths, max_length):
    """Independent batch-size-one replay: valid prefix per slot, zeros at padding."""
    n_seq = hidden.shape[1]
    obs_sequence = obs.reshape(n_seq, max_length, END2RACE_OBSERVATION_SIZE)
    start_sequence = episode_starts.reshape(n_seq, max_length)
    means = torch.zeros(n_seq, max_length, 2)
    hidden_by_step = torch.zeros(max_length, n_seq, stub.actor_hidden_size)
    final_hidden = torch.zeros_like(hidden)
    for slot in range(n_seq):
        h = hidden[:, slot : slot + 1]
        for step in range(max_length):
            h = h * (1.0 - start_sequence[slot, step])
            if step < lengths[slot]:
                step_obs = obs_sequence[slot, step]
                action, h = stub.end2race_actor(
                    step_obs[:END2RACE_LIDAR_SIZE].view(1, 1, -1),
                    step_obs[END2RACE_LIDAR_SIZE:].view(1, 1, -1),
                    h,
                )
                means[slot, step] = action[0, -1]
            hidden_by_step[step, slot] = h[0, 0]
        final_hidden[:, slot : slot + 1] = h
    actor_features = hidden_by_step.transpose(0, 1).reshape(-1, stub.actor_hidden_size)
    return means.reshape(-1, 2), final_hidden, actor_features


class BatchedReplayEquivalence(unittest.TestCase):
    def test_dtype_contract_fails_closed(self):
        stub = _make_stub()
        obs, hidden, starts = _make_inputs(2, 3)
        with self.assertRaisesRegex(RuntimeError, "actor input must be float32"):
            End2RaceGRUPolicy._actor_replay_batched(
                stub,
                obs.double(),
                (hidden, torch.zeros_like(hidden)),
                starts,
                _validity([3, 2], 3),
            )
        with self.assertRaisesRegex(RuntimeError, "transport tensors must be float32"):
            End2RaceGRUPolicy._actor_replay_batched(
                stub,
                obs,
                (hidden.double(), torch.zeros_like(hidden).double()),
                starts,
                _validity([3, 2], 3),
            )

    def test_validity_shape_contract_fails_closed(self):
        stub = _make_stub()
        obs, hidden, starts = _make_inputs(2, 3)
        with self.assertRaisesRegex(RuntimeError, "timestep count"):
            End2RaceGRUPolicy._actor_replay_batched(
                stub,
                obs,
                (hidden, torch.zeros_like(hidden)),
                starts,
                _validity([2, 2], 2),
            )
        with self.assertRaisesRegex(RuntimeError, "one entry per sequence"):
            End2RaceGRUPolicy._actor_replay_batched(
                stub,
                obs,
                (hidden, torch.zeros_like(hidden)),
                starts,
                ((True,), (True,), (False,)),
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the TF32 fail-closed test")
    def test_cuda_tf32_contract_fails_closed(self):
        stub = _make_stub()
        stub.end2race_actor.cuda()
        obs, hidden, starts = _make_inputs(1, 2)
        obs, hidden, starts = obs.cuda(), hidden.cuda(), starts.cuda()
        previous_cudnn = torch.backends.cudnn.allow_tf32
        previous_matmul = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = False
            with self.assertRaisesRegex(RuntimeError, "TF32 disabled"):
                End2RaceGRUPolicy._actor_replay_batched(
                    stub,
                    obs,
                    (hidden, torch.zeros_like(hidden)),
                    starts,
                    None,
                )
        finally:
            torch.backends.cudnn.allow_tf32 = previous_cudnn
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul

    def test_padded_layout_matches_batch1_reference(self):
        stub = _make_stub()
        n_seq, max_length = 3, 5
        lengths = [5, 3, 1]
        obs, hidden, starts = _make_inputs(n_seq, max_length)
        with torch.no_grad():
            ref_means, ref_final, ref_features = _reference_replay(
                stub, obs, hidden, starts, lengths, max_length
            )
            means, (final, _), features = End2RaceGRUPolicy._actor_replay_batched(
                stub, obs, (hidden, torch.zeros_like(hidden)), starts,
                _validity(lengths, max_length),
            )
        self.assertLess((means - ref_means).abs().max().item(), 1e-5)
        self.assertLess((final - ref_final).abs().max().item(), 1e-5)
        self.assertLess((features - ref_features).abs().max().item(), 1e-5)

    def test_invalid_positions_zero_mean_and_carried_hidden(self):
        stub = _make_stub()
        n_seq, max_length = 3, 5
        lengths = [5, 3, 1]
        obs, hidden, starts = _make_inputs(n_seq, max_length)
        processed_rows = []
        original = stub.end2race_actor.forward

        def counting_forward(lidar, speed, h):
            processed_rows.append(lidar.shape[0])
            return original(lidar, speed, h)

        stub.end2race_actor.forward = counting_forward
        with torch.no_grad():
            means, _, features = End2RaceGRUPolicy._actor_replay_batched(
                stub, obs, (hidden, torch.zeros_like(hidden)), starts,
                _validity(lengths, max_length),
            )
        stub.end2race_actor.forward = original
        # The actor processed exactly the valid positions and nothing else.
        self.assertEqual(sum(processed_rows), sum(lengths))
        means_grid = means.reshape(n_seq, max_length, 2)
        features_grid = features.reshape(n_seq, max_length, stub.actor_hidden_size)
        for slot, length in enumerate(lengths):
            for step in range(length, max_length):
                self.assertTrue(torch.equal(means_grid[slot, step], torch.zeros(2)))
                # Hidden at padded steps is carried bitwise unchanged.
                self.assertTrue(
                    torch.equal(features_grid[slot, step], features_grid[slot, length - 1])
                )

    def test_episode_start_resets_initial_hidden(self):
        stub = _make_stub()
        n_seq, max_length = 2, 3
        lengths = [3, 3]
        obs, hidden, starts = _make_inputs(n_seq, max_length)
        starts = starts.reshape(n_seq, max_length)
        starts[1, 0] = 1.0
        zeroed = hidden.clone()
        zeroed[:, 1] = 0.0
        with torch.no_grad():
            with_reset = End2RaceGRUPolicy._actor_replay_batched(
                stub, obs, (hidden, torch.zeros_like(hidden)), starts.reshape(-1),
                _validity(lengths, max_length),
            )
            with_zero_init = End2RaceGRUPolicy._actor_replay_batched(
                stub, obs, (zeroed, torch.zeros_like(zeroed)),
                torch.zeros(n_seq * max_length), _validity(lengths, max_length),
            )
        self.assertTrue(torch.equal(with_reset[0], with_zero_init[0]))
        self.assertTrue(torch.equal(with_reset[2], with_zero_init[2]))

    def test_single_sequence_bitwise_matches_collection_path(self):
        stub = _make_stub()
        obs, hidden, starts = _make_inputs(1, 4)
        with torch.no_grad():
            collection = End2RaceGRUPolicy._actor_forward(
                stub, obs, (hidden, torch.zeros_like(hidden)), starts
            )
            replay = End2RaceGRUPolicy._actor_replay_batched(
                stub, obs, (hidden, torch.zeros_like(hidden)), starts, None
            )
        # n_seq == 1 uses identical kernel shapes, so the paths agree bitwise.
        self.assertTrue(torch.equal(collection[0], replay[0]))
        self.assertTrue(torch.equal(collection[2], replay[2]))

    def test_masked_gradients_match_batch1_reference(self):
        n_seq, max_length = 3, 4
        lengths = [4, 2, 1]
        validity = _validity(lengths, max_length)
        mask = torch.tensor(
            [step < lengths[slot] for slot in range(n_seq) for step in range(max_length)]
        )
        target = torch.randn(n_seq * max_length, 2, generator=torch.Generator().manual_seed(3))

        def masked_loss_grads(run):
            stub = _make_stub()
            obs, hidden, starts = _make_inputs(n_seq, max_length)
            means = run(stub, obs, hidden, starts)
            loss = ((means - target)[mask] ** 2).mean()
            loss.backward()
            grads = torch.cat(
                [p.grad.reshape(-1) for p in stub.end2race_actor.parameters() if p.grad is not None]
            )
            return loss.detach(), grads.double()

        loss_ref, grads_ref = masked_loss_grads(
            lambda stub, obs, hidden, starts: _reference_replay(
                stub, obs, hidden, starts, lengths, max_length
            )[0]
        )
        loss_b, grads_b = masked_loss_grads(
            lambda stub, obs, hidden, starts: End2RaceGRUPolicy._actor_replay_batched(
                stub, obs, (hidden, torch.zeros_like(hidden)), starts, validity
            )[0]
        )
        self.assertLess(abs(loss_b.item() - loss_ref.item()), 1e-6)
        cosine = torch.dot(grads_ref, grads_b) / (grads_ref.norm() * grads_b.norm())
        self.assertGreaterEqual(cosine.item(), 0.999999)
        relative_l2 = (grads_b - grads_ref).norm() / grads_ref.norm()
        self.assertLessEqual(relative_l2.item(), 1e-4)


if __name__ == "__main__":
    unittest.main()
