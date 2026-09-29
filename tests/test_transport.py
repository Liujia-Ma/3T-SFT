import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
from three_t_sft.transport import (token_ste, receiver_ste, manual_proxy,
                                  Bridges, final_answer_loss, policy_kl, consistency_loss)
from three_t_sft.alignment import align, Span


class TransportTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(42)

    def test_hard_forward_soft_backward(self):
        z = torch.randn(3, 7, dtype=torch.double, requires_grad=True)
        e = torch.randn(7, 4, dtype=torch.double)
        ids = torch.tensor([2, 1, 6])
        st = token_ste(z, ids, e)
        self.assertTrue(torch.equal(st, e[ids]))
        actual, = torch.autograd.grad(st.sum(), z)
        expected, = torch.autograd.grad((z.softmax(-1)@e).sum(), z)
        torch.testing.assert_close(actual, expected)

    def test_manual_vjp_matches_explicit_second_ste(self):
        z = torch.randn(3, 7, dtype=torch.double, requires_grad=True)
        e = torch.randn(7, 4, dtype=torch.double)
        bridge = torch.nn.Linear(4, 5).double()
        a = torch.tensor([[.5, .5, 0], [0, 0, 1]], dtype=torch.double)
        c = a@bridge(token_ste(z, torch.tensor([1, 2, 3]), e))
        real = torch.randn(2, 5, dtype=torch.double)
        st = receiver_ste(real, c, mu=.7)
        self.assertTrue(torch.equal(st, real))
        objective = st.sin().square().sum()
        g, = torch.autograd.grad(objective, st, retain_graph=True)
        params = (z, *bridge.parameters())
        expected = torch.autograd.grad(objective, params, retain_graph=True)
        actual = torch.autograd.grad(manual_proxy(c, g, mu=.7), params)
        for x, y in zip(actual, expected): torch.testing.assert_close(x, y)

    def test_equal_dimensions_do_not_imply_identity(self):
        b = Bridges()
        b.register('a', 'b', 4, 4)
        b.register('b', 'c', 4, 4, shared_space=True)
        self.assertIsInstance(b.layers['pair_0'], torch.nn.Linear)
        self.assertIsInstance(b.layers['pair_1'], torch.nn.Identity)

    def test_per_example_loss_and_masking(self):
        logits = torch.randn(2, 3, 5, requires_grad=True)
        labels = torch.tensor([[1, -100, -100], [2, 3, 4]])
        loss = final_answer_loss(logits, labels)
        expected = (torch.nn.functional.cross_entropy(logits[0, :1], labels[0, :1])+
                    torch.nn.functional.cross_entropy(logits[1], labels[1]))/2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(logits.grad[0, 1:].abs().sum().item(), 0)

    def test_reference_kl_is_detached(self):
        z = torch.randn(3, 5, requires_grad=True)
        ref = torch.randn(3, 5, requires_grad=True)
        policy_kl(z, ref, torch.tensor([True, False, True])).backward()
        self.assertIsNone(ref.grad)
        self.assertEqual(z.grad[1].abs().sum().item(), 0)

    def test_consistency_masks_unsupported_rows(self):
        surrogate = torch.randn(3, 4, requires_grad=True)
        real = torch.randn(3, 4, requires_grad=True)
        consistency_loss(surrogate, real, (0, 2)).backward()
        self.assertIsNone(real.grad)
        self.assertEqual(surrogate.grad[1].abs().sum().item(), 0)
        self.assertGreater(surrogate.grad[0].norm().item(), 0)

    def test_empty_alignment_tensor(self):
        mapping = align((), (), Span(0, 0))
        sparse = mapping.tensor(dtype=torch.double)
        self.assertEqual(tuple(sparse.shape), (0, 0))
        self.assertEqual(sparse._nnz(), 0)


if __name__ == '__main__': unittest.main()
