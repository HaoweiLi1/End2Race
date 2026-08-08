import unittest
import torch
from average_actor_checkpoints import average_state_dicts


class TestAverageActorCheckpoints(unittest.TestCase):

    def test_float_tensors_use_equal_average_and_preserve_dtype(self):
        state_dicts = []
        for value in (1.0, 2.0, 3.0, 8.0):
            state_dicts.append({"weight": torch.tensor([value, value + 1.0], dtype=torch.float32)})
        averaged = average_state_dicts(state_dicts)
        self.assertEqual(averaged["weight"].dtype, torch.float32)
        torch.testing.assert_close(averaged["weight"], torch.tensor([3.5, 4.5], dtype=torch.float32), rtol=0, atol=0)

    def test_non_floating_tensor_must_match(self):
        state_dicts = [{"count": torch.tensor([1], dtype=torch.int64)} for _ in range(4)]
        state_dicts[-1]["count"][0] = 2
        with self.assertRaisesRegex(RuntimeError, "Non-floating actor tensor differs"):
            average_state_dicts(state_dicts)

    def test_key_order_shape_and_source_count_fail_closed(self):
        valid = {"weight": torch.zeros(2)}
        with self.assertRaisesRegex(RuntimeError, "Exactly four"):
            average_state_dicts([valid, valid, valid])
        with self.assertRaisesRegex(RuntimeError, "keys or order differ"):
            average_state_dicts([valid, valid, valid, {"other": torch.zeros(2)}])
        with self.assertRaisesRegex(RuntimeError, "tensor contract differs"):
            average_state_dicts([valid, valid, valid, {"weight": torch.zeros(3)}])


if __name__ == "__main__":
    unittest.main()
