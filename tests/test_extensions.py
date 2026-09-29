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

    def test_answer_label_masking(self):
        import torch
        from three_t_sft.transport import answer_labels
        ids = torch.tensor([[1, 2, 3, 0]])
        labels = answer_labels(ids, torch.tensor([[False, True, True, True]]), torch.tensor([[1, 1, 1, 0]]))
        self.assertEqual(labels.tolist(), [[-100, 2, 3, -100]])
        self.assertEqual(ids.tolist(), [[1, 2, 3, 0]])

    def test_paired_seed_statistics(self):
        from three_t_sft.metrics import paired_gains, summarize_seeds
        r = paired_gains({42: 3., 43: 5.}, {42: 2., 43: 3.})
        self.assertEqual(r['mean'], 1.5)
        self.assertIsNone(summarize_seeds([1.])['sample_std'])
        with self.assertRaises(ValueError): paired_gains({1: 2.}, {2: 2.})

    def test_final_answer_loader_drops_rationales(self):
        import tempfile, json
        from three_t_sft.data import load_final_answers
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'input.jsonl'
            p.write_text(json.dumps({'id':'x','question':'q','answer':'a','rationale':'unused'}))
            self.assertEqual(set(load_final_answers(p)[0]), {'id','question','answer'})
            p.write_text(p.read_text()+'\n'+p.read_text())
            with self.assertRaises(ValueError): load_final_answers(p)
