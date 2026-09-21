import importlib.util
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('terrain_eval_metrics', Path(__file__).parents[1] / 'terrain_eval_metrics.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_supported_crossing_and_failure_precedence():
    pos = torch.tensor([[3.6, 0.], [3.6, 0.], [3.6, 1.1], [0., 0.], [3.6, 0.], [3.6, 0.], [4., 0.]])
    contact = torch.tensor([False, True, False, False, False, False, False])
    timeout = torch.tensor([False, False, False, True, False, False, False])
    support = torch.tensor([True, True, True, True, False, True, True])
    upright = torch.tensor([True, True, True, True, True, False, True])
    assert m.traversal_outcome(pos, contact, timeout, support, upright).tolist() == [1, 2, 3, 4, 0, 0, 3]


def test_arrival_on_deadline_is_success_only_if_supported():
    pos = torch.tensor([[3.5, 1.0], [3.5, 0.]])
    assert m.traversal_outcome(pos, torch.zeros(2,dtype=torch.bool), torch.ones(2,dtype=torch.bool), torch.tensor([True,False]), torch.ones(2,dtype=torch.bool)).tolist() == [1,4]
