"""Reverse-topological manual VJP on a fixed realized communication trace."""
from dataclasses import dataclass
from math import isfinite
from typing import Callable
import torch
from .transport import manual_proxy


@dataclass
class Computation:
    # Outgoing aligned surrogate slices, keyed by edge ID.
    outgoing: dict[str, torch.Tensor]
    final_loss: torch.Tensor | None = None
    local_stabilizer: torch.Tensor | None = None


@dataclass
class Node:
    id: str
    recompute: Callable[[dict[str, torch.Tensor]], Computation]


@dataclass
class Edge:
    id: str
    sender: str
    receiver: str
    real: torch.Tensor


def backward_trace(nodes, edges, terminal, *, mu=1.0, loss_scale=1.0):
    """Accumulate parameter gradients; caller owns zero_grad/step.

    recompute must replay the same recorded hard prefixes and RNG state. All
    incoming real slices become independent leaves. Only task-induced gradients
    are transported upstream; local stabilizers never enter recursive credit.
    loss_scale supports batch/Monte Carlo averaging, applied exactly once at the
    terminal and once per local stabilizer, not at every message hop.
    """
    if not isfinite(loss_scale) or loss_scale <= 0 or not isfinite(mu) or mu <= 0:
        raise ValueError('Finite positive loss and upstream scales required')
    by_id = {n.id: n for n in nodes}
    if len(by_id) != len(nodes) or terminal not in by_id:
        raise ValueError('Unique nodes and a known terminal are required')
    if len({e.id for e in edges}) != len(edges): raise ValueError('Duplicate edge ID')
    incoming = {n.id: [] for n in nodes}
    outgoing = {n.id: [] for n in nodes}
    for edge in edges:
        if edge.sender not in by_id or edge.receiver not in by_id or edge.real.ndim != 2:
            raise ValueError('Invalid message edge')
        incoming[edge.receiver].append(edge); outgoing[edge.sender].append(edge)
    pending = {k: len(v) for k, v in incoming.items()}
    queue = [k for k, v in pending.items() if v == 0]
    order = []
    while queue:
        key = queue.pop(0); order.append(key)
        for edge in outgoing[key]:
            pending[edge.receiver] -= 1
            if pending[edge.receiver] == 0: queue.append(edge.receiver)
    if len(order) != len(nodes): raise ValueError('Realized trace must be acyclic')
    ancestors = {terminal}
    for key in reversed(order):
        if key in ancestors: ancestors.update(e.sender for e in incoming[key])
    gradients, report = {}, {'visited_nodes': [], 'transport_norms': {}}
    for key in reversed(order):
        if key not in ancestors: continue
        leaves = {e.id: e.real.detach().clone().requires_grad_(True) for e in incoming[key]}
        computation = by_id[key].recompute(leaves)
        terms = []
        if key == terminal:
            if computation.final_loss is None: raise ValueError('Terminal must supply a final-answer loss')
            terms.append(computation.final_loss*loss_scale)
            report['final_loss'] = computation.final_loss.detach().item()
        for edge in outgoing[key]:
            if edge.id in gradients:
                if edge.id not in computation.outgoing: raise ValueError('Missing outgoing surrogate')
                terms.append(manual_proxy(computation.outgoing[edge.id], gradients[edge.id], mu))
        task = sum(terms) if terms else None
        if task is not None and task.requires_grad and leaves:
            captured = torch.autograd.grad(task, tuple(leaves.values()), retain_graph=True, allow_unused=True)
        else: captured = [None]*len(leaves)
        for (edge_id, leaf), grad in zip(leaves.items(), captured):
            gradients[edge_id] = torch.zeros_like(leaf) if grad is None else grad.detach()
            report['transport_norms'][edge_id] = gradients[edge_id].norm().item()
        local = computation.local_stabilizer
        objective = task
        if local is not None: objective = local*loss_scale if task is None else task+local*loss_scale
        if objective is not None and objective.requires_grad: objective.backward()
        report['visited_nodes'].append(key)
    return report
