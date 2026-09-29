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

    def test_atomic_checkpoint_roundtrip(self):
        import torch, tempfile
        from three_t_sft.checkpoint import atomic_save
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'best.pt'
            atomic_save({'weight':torch.tensor([1.])}, p)
            atomic_save({'weight':torch.tensor([2.])}, p)
            self.assertEqual(torch.load(p, weights_only=True)['weight'].item(), 2.)
            self.assertEqual(len(list(Path(folder).iterdir())), 1)

    def test_parameter_counts_respect_freezing(self):
        import torch
        from three_t_sft.metrics import parameter_counts
        m = torch.nn.Linear(3, 2)
        m.weight.requires_grad_(False)
        self.assertEqual(parameter_counts(m), {'total':8,'trainable':2,'frozen':6})

    def test_evaluation_uses_one_trace_per_example(self):
        import torch
        from unittest.mock import patch
        from three_t_sft.tiny import TinyWorkflow
        from three_t_sft.train import evaluate
        torch.set_num_threads(1)
        model = TinyWorkflow()
        with patch.object(model, 'trace', wraps=model.trace) as trace:
            result = evaluate(model, [{'question':'1+2=','answer':'3'}], 2)
            self.assertEqual(trace.call_count, 1)
        self.assertEqual(result['count'], 1)

    def test_cli_config_overrides_preserve_source(self):
        import json
        from three_t_sft.config import apply_overrides
        base = json.loads((Path(__file__).resolve().parents[1]/'configs/tiny.json').read_text())
        out = apply_overrides(base, steps=2, seed=7)
        self.assertEqual(out['steps'], 2)
        self.assertEqual(out['seed'], 7)
        self.assertEqual(base['steps'], 30)

    def test_stochastic_node_callback_repeats_forward(self):
        import torch
        from three_t_sft.reproducibility import RNGState, replayable_node
        state = RNGState.capture()
        node = replayable_node('dropout', lambda x: torch.nn.functional.dropout(x['value'],.5,training=True), state)
        one = node.recompute({'value':torch.ones(100)})
        two = node.recompute({'value':torch.ones(100)})
        self.assertTrue(torch.equal(one,two))

    def test_global_gradient_norm_and_nonfinite_failure(self):
        import torch
        from three_t_sft.metrics import gradient_norm
        p = torch.nn.Parameter(torch.zeros(2)); p.grad = torch.tensor([3.,4.])
        self.assertEqual(gradient_norm([p]),5.)
        p.grad[0] = float('nan')
        with self.assertRaises(ValueError): gradient_norm([p])

    def test_existing_experiment_is_preserved(self):
        import tempfile,json
        from three_t_sft.train import run
        config = json.loads((Path(__file__).resolve().parents[1]/'configs/tiny.json').read_text())
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)/'metrics.jsonl'; p.write_text('keep')
            with self.assertRaises(FileExistsError): run(config,folder)
            self.assertEqual(p.read_text(),'keep')
