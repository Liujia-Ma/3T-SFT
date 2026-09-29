import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import torch
from three_t_sft.engine import Computation, Node, Edge, backward_trace


class EngineTests(unittest.TestCase):
    def test_recursive_credit_and_local_stabilizer_isolation(self):
        a = torch.tensor(2., requires_grad=True)
        b = torch.tensor(3., requires_grad=True)
        real1, real2 = torch.tensor([[4.]]), torch.tensor([[12.]])
        nodes = [Node('a', lambda x: Computation({'ab': a.reshape(1, 1)*2})),
                 Node('b', lambda x: Computation({'bc': x['ab']*b},
                     local_stabilizer=(x['ab']*b).square().sum())),
                 Node('c', lambda x: Computation({}, final_loss=x['bc'].square().sum()))]
        report = backward_trace(nodes, [Edge('ab', 'a', 'b', real1), Edge('bc', 'b', 'c', real2)], 'c')
        # Task: 12^2 -> d/da = 24*3*2. Local b penalty must not double a's gradient.
        self.assertAlmostEqual(a.grad.item(), 144.)
        self.assertAlmostEqual(b.grad.item(), 192.)
        self.assertEqual(report['visited_nodes'], ['c', 'b', 'a'])

    def test_branch_accumulation_and_unrealized_exclusion(self):
        a = torch.tensor(2., requires_grad=True)
        dead = torch.tensor(5., requires_grad=True)
        nodes = [Node('a', lambda _: Computation({'ab': a.reshape(1, 1), 'ac': a.reshape(1, 1)})),
                 Node('b', lambda x: Computation({'bt': 2*x['ab']})),
                 Node('c', lambda x: Computation({'ct': 3*x['ac']})),
                 Node('t', lambda x: Computation({}, final_loss=(x['bt']+x['ct']).sum())),
                 Node('dead', lambda _: Computation({}, local_stabilizer=dead.square()))]
        one = torch.ones(1, 1)
        edges = [Edge('ab', 'a', 'b', one), Edge('ac', 'a', 'c', one),
                 Edge('bt', 'b', 't', one), Edge('ct', 'c', 't', one)]
        backward_trace(nodes, edges, 't', loss_scale=.5)
        self.assertEqual(a.grad.item(), 2.5)
        self.assertIsNone(dead.grad)

    def test_cycle_rejected(self):
        nodes = [Node('a', None), Node('b', None)]
        edges = [Edge('ab', 'a', 'b', torch.ones(1, 1)), Edge('ba', 'b', 'a', torch.ones(1, 1))]
        with self.assertRaises(ValueError): backward_trace(nodes, edges, 'b')


if __name__ == '__main__': unittest.main()
