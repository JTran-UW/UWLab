import math
import sys

import torch

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
from success_guided_sampler import SuccessGuidedSampler  # noqa: E402


def kernel_ref(p, t=0.5, k=1.0, T=2.0, eps=1e-4):
    w = max((p + eps) ** (k * t) * (1 - p + eps) ** (k * (1 - t)), eps)
    return math.exp(math.log(w + eps) / T)


def test_kernel_matches_paper():
    s = SuccessGuidedSampler(4, history=10)
    for i, n_succ in enumerate([0, 5, 10]):
        s.update(torch.full((10,), i), torch.arange(10) < n_succ)
    ref = [kernel_ref(0.0), kernel_ref(0.5), kernel_ref(1.0), kernel_ref(0.5)]  # state 3 unvisited -> empty_rate 0.5
    ref = torch.tensor(ref) / sum(ref)
    assert torch.allclose(s.probs(), ref, atol=1e-6), (s.probs(), ref)
    r = s.probs()[1] / s.probs()[0]
    assert 6.5 < r < 7.5, r  # p=0.5 vs p=0 under t .5 k 1 T 2


def test_mean_over_filled_and_empty():
    s = SuccessGuidedSampler(3, history=100, empty_rate=0.5)
    s.update(torch.tensor([0, 0, 1]), torch.tensor([True, False, True]))
    p = s.success_rate()
    assert torch.allclose(p, torch.tensor([0.5, 1.0, 0.5])), p


def test_duplicates_in_one_batch_get_own_slots():
    s = SuccessGuidedSampler(2, history=8)
    ids = torch.tensor([1, 0, 1, 1, 0, 1])
    out = torch.tensor([True, False, False, True, False, True])
    s.update(ids, out)
    assert s.count.tolist() == [2, 4]
    assert s.outcomes[1, :4].sum().item() == 3 and s.outcomes[1, 4:].sum().item() == 0
    assert abs(s.success_rate()[1].item() - 0.75) < 1e-6


def test_more_duplicates_than_history_keeps_latest():
    s = SuccessGuidedSampler(1, history=4)
    s.update(torch.zeros(10, dtype=torch.long), torch.tensor([True] * 6 + [False] * 4))  # last 4 are failures
    assert s.success_rate()[0].item() == 0.0, s.success_rate()


def test_ring_wraps_oldest_first():
    s = SuccessGuidedSampler(1, history=4)
    for v in [True, True, True, True]:
        s.update(torch.tensor([0]), torch.tensor([v]))
    for v in [False, False]:
        s.update(torch.tensor([0]), torch.tensor([v]))
    assert abs(s.success_rate()[0].item() - 0.5) < 1e-6


def test_base_measure_and_sampling_frequency():
    base = torch.tensor([3.0, 1.0])
    s = SuccessGuidedSampler(2, base_weights=base)
    assert torch.allclose(s.probs(), torch.tensor([0.75, 0.25]))
    torch.manual_seed(0)
    draws = s.sample(200_000)
    frac = (draws == 0).float().mean().item()
    assert abs(frac - 0.75) < 0.01, frac


def negative_controls():
    """Each mutation must make at least one test fail."""
    import success_guided_sampler as m

    orig_update, orig_rate = m.SuccessGuidedSampler.update, m.SuccessGuidedSampler.success_rate

    def naive_update(self, ids, out):  # duplicates collide on one slot
        slot = self.count[ids] % self.history
        self.outcomes[ids, slot] = out.bool()
        self.count.index_add_(0, ids, torch.ones_like(ids))

    def mean_over_H(self):  # option A instead of B
        return self.outcomes.float().mean(dim=1)

    failures = []
    for name, attr, fn in [("naive_update", "update", naive_update), ("mean_over_H", "success_rate", mean_over_H)]:
        setattr(m.SuccessGuidedSampler, attr, fn)
        caught = False
        for t in TESTS:
            try:
                t()
            except AssertionError:
                caught = True
                break
        setattr(m.SuccessGuidedSampler, "update", orig_update)
        setattr(m.SuccessGuidedSampler, "success_rate", orig_rate)
        print(f"negative control {name}: {'caught' if caught else 'NOT CAUGHT'}")
        failures += [] if caught else [name]
    return failures


TESTS = [
    test_kernel_matches_paper,
    test_mean_over_filled_and_empty,
    test_duplicates_in_one_batch_get_own_slots,
    test_more_duplicates_than_history_keeps_latest,
    test_ring_wraps_oldest_first,
    test_base_measure_and_sampling_frequency,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS {t.__name__}")
    bad = negative_controls()
    sys.exit(1 if bad else 0)
