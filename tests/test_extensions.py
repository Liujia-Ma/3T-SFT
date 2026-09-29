import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

class ExtensionTests(unittest.TestCase):

    def test_nonfinite_trace_scales(self):
        from three_t_sft.engine import backward_trace
        for bad in (float('nan'), float('inf'), -1.):
            with self.assertRaises(ValueError): backward_trace([], [], 'terminal', loss_scale=bad)
            with self.assertRaises(ValueError): backward_trace([], [], 'terminal', mu=bad)

    def test_invalid_message_tensors(self):
        import torch
        from three_t_sft.engine import Node, Edge, backward_trace
        for tensor in (torch.ones(1, 2, dtype=torch.long), torch.tensor([[float('nan')]])):
            with self.assertRaises(ValueError):
                backward_trace([Node('a', None), Node('b', None)], [Edge('e', 'a', 'b', tensor)], 'b')

    def test_invalid_hard_tokens(self):
        import torch
        from three_t_sft.transport import token_ste
        for ids in (torch.tensor([-1]), torch.tensor([3]), torch.tensor([1.])):
            with self.assertRaises(ValueError): token_ste(torch.zeros(1, 3), ids, torch.zeros(3, 2))

    def test_training_config_validation(self):
        import json
        from three_t_sft.config import validate_config
        base = json.loads((Path(__file__).resolve().parents[1]/'configs/tiny.json').read_text())
        for key, value in [('steps', True), ('learning_rate', float('nan')), ('mode', 'unknown')]:
            with self.assertRaises(ValueError): validate_config(dict(base, **{key: value}))
        self.assertEqual(validate_config(base), base)

    def test_replay_rng_restores_caller(self):
        import torch
        from three_t_sft.reproducibility import RNGState, replay_rng
        torch.manual_seed(3)
        state = RNGState.capture()
        expected = torch.rand(4)
        caller = torch.get_rng_state().clone()
        with replay_rng(state): torch.testing.assert_close(torch.rand(4), expected)
        self.assertTrue(torch.equal(caller, torch.get_rng_state()))
