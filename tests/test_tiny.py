import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
from three_t_sft.tiny import TinyWorkflow, TextTokenizer
from three_t_sft.train import data


class TinyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1); torch.manual_seed(42)

    def test_actual_retokenization(self):
        a, b = TextTokenizer(), TextTokenizer(True)
        ids, _ = a.encode('12+3=')
        text, _ = a.decode_with_spans(ids)
        other, _ = b.encode(text)
        self.assertNotEqual(len(ids), len(other))
        self.assertEqual(b.decode_with_spans(other)[0], text)

    def test_upstream_adapters_get_final_answer_gradient(self):
        model = TinyWorkflow()
        model.backward_example('2+3=', '5')
        for agent in model.agents:
            self.assertGreater(agent.lora_b.grad.norm().item(), 0)
            self.assertIsNone(agent.embedding.weight.grad)
        self.assertTrue(any(p.grad is not None and p.grad.norm() > 0 for p in model.bridges.parameters()))

    def test_terminal_baseline_has_no_upstream_gradients(self):
        model = TinyWorkflow()
        model.backward_example('2+3=', '5', mode='terminal')
        for agent in model.agents[:2]:
            self.assertIsNone(agent.lora_b.grad)
        self.assertGreater(model.agents[-1].lora_b.grad.norm().item(), 0)

    def test_inference_does_not_use_bridges(self):
        model = TinyWorkflow()
        expected = model.predict('2+3=')
        def fail(*args): raise AssertionError('Deployment touched a training-only bridge')
        model.bridges.forward = fail
        self.assertEqual(model.predict('2+3='), expected)

    def test_homogeneous_identity(self):
        model = TinyWorkflow(heterogeneous=False)
        self.assertEqual(sum(p.numel() for p in model.bridges.parameters()), 0)
        model.backward_example('2+3=', '5')
        self.assertGreater(model.agents[0].lora_b.grad.norm().item(), 0)

    def test_split_ids_are_disjoint(self):
        a, b, c = [{x['id'] for x in split} for split in data(42)]
        self.assertEqual(len(a|b|c), 100)
        self.assertFalse(a&b or a&c or b&c)


if __name__ == '__main__': unittest.main()
