"""Exact interventional Shapley explanations for the deployed hybrid model."""
from math import comb
import numpy as np

from hybrid_model import hybrid_probabilities


def exact_shapley(tree, lstm, sequence, blend_weight, baselines, selected_class):
    """Return exact per-feature Shapley values and endpoint probabilities.

    A feature is treated as one player across every timestep. Because this project
    has 11 players, all 2^11 coalitions can be evaluated without approximation.
    """
    sequence = np.asarray(sequence, dtype=float)
    m = sequence.shape[2]
    masks = ((np.arange(2**m)[:, None] >> np.arange(m)) & 1).astype(bool)
    baseline = np.broadcast_to(np.asarray(baselines), sequence.shape)
    coalitions = np.where(masks[:, None, :], sequence, baseline)
    values = hybrid_probabilities(tree, lstm, coalitions, blend_weight)[:, selected_class]
    phi = np.zeros(m)
    mask_ids = np.arange(2**m)
    for feature in range(m):
        absent = mask_ids[(mask_ids & (1 << feature)) == 0]
        sizes = masks[absent].sum(axis=1)
        weights = np.array([1.0 / (m * comb(m - 1, int(size))) for size in sizes])
        phi[feature] = np.sum(weights * (values[absent | (1 << feature)] - values[absent]))
    return phi, float(values[0]), float(values[-1])
