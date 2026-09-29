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
