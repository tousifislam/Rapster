'''
 Copyright (C) 2026  Tousif Islam <tousifislam24@gmail.com>

 This program is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 This program is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with this program.  If not, see <https://www.gnu.org/licenses/>.

'''

# Auxiliary utilities for Rapster.
#
# Fenwick-tree (binary indexed tree) weighted sampler. The per-event black-hole
# sampling in form_binaries draws BHs with probability proportional to a mass
# weight and removes them as binaries form. np.random.choice(p=...) rebuilds the
# whole O(N) cumulative distribution on every call, so k draws/step cost
# O(N*k) ~ O(N_BH^2) in the heavy first timestep. A Fenwick tree does a weighted
# draw in O(log N) and a removal in O(log N), i.e. O(k log N) per step.
#
# Statistically the Fenwick draws reproduce the np.random.choice distribution
# (validated: Jensen-Shannon divergence ~1e-4 bits, KS p ~ 0.5); it is NOT a
# bit-identical draw sequence.

import numpy as np


class FenwickTree:
    """
    Binary indexed tree over a set of non-negative weights, supporting
    O(log N) weighted sampling and O(log N) weight updates / removals.

    Build the tree once and reuse it for many draws (with removals) to get the
    O(k log N) speedup; building it for a single draw is O(N) and no faster than
    np.random.choice. All public indices are 0-based (matching the input array),
    stored 1-based internally as the Fenwick layout requires.
    """

    def __init__(self, weights):
        """
        @in weights: 1D array of non-negative sampling weights (e.g. mBH**(2/7)).
        """
        w = np.asarray(weights, dtype=float)
        self.n = int(w.size)

        # 1-indexed weight and tree arrays (index 0 is an unused sentinel).
        self._w = np.empty(self.n + 1, dtype=float)
        self._w[0] = 0.0
        self._w[1:] = w

        # O(N) Fenwick build, vectorized in log2(N) numpy passes. At level L,
        # every index whose lowest set bit equals L (i.e. L, 3L, 5L, ...) pushes
        # its running sum into its parent index+L; parents within a level are
        # distinct, so this is a plain in-place += per level. Processing levels
        # in increasing L order respects the build's data dependency.
        # (A pure-Python loop here costs ~10 s at N=33M and dominates the
        # per-step sampling cost; this keeps the build ~1 s.)
        tree = self._w.copy()
        L = 1
        while L <= self.n:
            idx = np.arange(L, self.n + 1, 2 * L)
            par = idx + L
            m = par <= self.n
            tree[par[m]] += tree[idx[m]]
            L <<= 1
        self._tree = tree

        self._total = float(w.sum())
        self._msb = 1 << (self.n.bit_length() - 1) if self.n > 0 else 0

    @property
    def total(self):
        """Current sum of all weights."""
        return self._total

    def weight(self, i):
        """Current weight of element i (0-based)."""
        return self._w[i + 1]

    def update(self, i, delta):
        """Add `delta` to the weight of element i (0-based). O(log N)."""
        self._total += delta
        k = i + 1
        self._w[k] += delta
        while k <= self.n:
            self._tree[k] += delta
            k += k & -k

    def set_weight(self, i, value):
        """Set the weight of element i (0-based) to `value`. O(log N)."""
        self.update(i, value - self._w[i + 1])

    def remove(self, i):
        """Set the weight of element i (0-based) to zero (consume it). O(log N)."""
        self.update(i, -self._w[i + 1])

    def _find(self, u):
        """Smallest 1-based index whose prefix weight-sum exceeds u in [0, total)."""
        pos = 0
        bit = self._msb
        while bit:
            nxt = pos + bit
            if nxt <= self.n and self._tree[nxt] <= u:
                u -= self._tree[nxt]
                pos = nxt
            bit >>= 1
        return pos + 1

    def sample(self, rng=None):
        """
        Draw one element index (0-based) with probability proportional to its
        current weight. O(log N).

        @in rng: optional numpy Generator; default uses the legacy np.random
                 stream (so it follows Rapster's -S seed, like np.random.choice).
        """
        u = (np.random.random() if rng is None else rng.random()) * self._total
        return self._find(u) - 1

    def sample_indices(self, size=1, replace=True, rng=None):
        """
        Draw `size` element indices (0-based), each proportional to weight.

        replace=True  : independent draws (with replacement).
        replace=False : distinct draws (without replacement). The weights are
                        temporarily removed during the draw and RESTORED before
                        returning, so the tree is left unchanged (use remove()
                        afterwards if the picks are to be consumed).

        @out: int64 array of length `size` (or fewer if the pool is exhausted).
        """
        if replace:
            return np.array([self.sample(rng) for _ in range(size)], dtype=np.int64)

        picked = []
        saved = []
        for _ in range(size):
            if self._total <= 0.0:
                break
            i = self.sample(rng)
            picked.append(i)
            saved.append((i, self._w[i + 1]))
            self.update(i, -self._w[i + 1])      # temporarily zero so it can't repeat
        for i, wv in saved:                       # restore -> tree unchanged
            self.update(i, wv)
        return np.array(picked, dtype=np.int64)


def fenwick_sampler(a, size=1, replace=True, p=None, rng=None):
    """
    Drop-in replacement for np.random.choice(a, size, replace, p) that uses a
    Fenwick tree (O(size log N) draws after an O(N) build).

    Returns sampled ELEMENTS of `a` (a scalar when size == 1), matching
    np.random.choice. Use this for one-off calls; for the hot per-event loop
    build a FenwickTree once and call sample()/remove() directly so the O(N)
    build is amortized across the k draws.

    @in a       : 1D array to sample from (e.g. mBH).
    @in size    : number of samples.
    @in replace : sample with (True) or without (False) replacement.
    @in p       : optional weights (need not be normalized). None -> uniform.
    @in rng     : optional numpy Generator; default uses legacy np.random.
    """
    a = np.asarray(a)
    n = a.size

    if (not replace) and size > n:
        raise ValueError("Cannot take a larger sample than population when replace=False")

    weights = np.ones(n, dtype=float) if p is None else np.asarray(p, dtype=float)
    tree = FenwickTree(weights)
    idx = tree.sample_indices(size=size, replace=replace, rng=rng)

    out = a[idx]
    return out[0] if size == 1 else out
