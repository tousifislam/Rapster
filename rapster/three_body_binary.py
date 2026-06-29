'''
 Copyright (C) 2026  Konstantinos Kritos <kkritos1@jhu.edu>

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

from .constants import *
from .functions import *
from .auxiliary import FenwickTree

def fast_sample_3bodyBinary(mBH, n_samples=1, approx_mBH_sampling=False):
    """
    Fast sampling for using rejection sampling with complexity O(N).

    @in mBH: array of BH masses
    @in n_samples: number of samples (m1, m2, m3)
    @in approx_mBH_sampling: if True, skip the (unused) per-call proposal-weight
        build below. The actual sampling here is uniform with rejection on the
        interaction term, so the weights are dead computation; skipping them is
        bit-identical (no RNG is consumed by their construction).
    """

    if not approx_mBH_sampling:
        # Pre-calculate proposal weights
        # m1 and m2 are dominated by m^4
        weights_binary = mBH**4
        p_binary = weights_binary / weights_binary.sum()

        # m3 is dominated by m^2.5
        weights_third = mBH**2.5
        p_third = weights_third / weights_third.sum()

    # Define the 'Interaction' term for the rejection criteria
    # Since (m1+m2)^-0.5 and (m1+m2+m3)^-0.5 are DECREASING functions,
    # the maximum value occurs at the smallest possible masses.
    min_m = np.min(mBH)
    max_interaction_val = (min_m + min_m)**(-0.5) * (min_m + min_m + min_m)**(-0.5)
    
    results = []
    while len(results) < n_samples:
        # Sample candidates
        # We need 3 unique indices
        indices = np.random.choice(len(mBH), size=3, replace=False)
        m1, m2 = mBH[indices[0]], mBH[indices[1]]
        m3 = mBH[indices[2]]
        
        # Calculate the interaction/correction factor
        # P_actual / P_proposal_separable
        interaction_val = (m1 + m2)**(-0.5) * (m1 + m2 + m3)**(-0.5)
        
        # Accept/Reject
        if np.random.rand() < (interaction_val / max_interaction_val):
            results.append((m1, m2, m3))
            
    return results[0] if n_samples == 1 else results

def three_body_binary(t, z, k_3bb, mBH_avg, binaries, mBH, sBH, gBH, hBH, vBH, N_3bb, N_BBH, random_pairing=False, approx_mBH_sampling=False):
    """
    @in t: simulation time
    @in z: simulation redshift
    @in k_3bb: number of 3bbs in current step
    @in mBH_avg: average BH mass
    @in binaries: array of BBHs
    @in mBH: array of single BH masses
    @in sBH: array of single BH spins
    @in gBH: array of single BH generations
    @in hBH: array of BH tdes count
    @in vBH: 3D BH velocity dispersion
    @in N_3bb: number of 3bbs
    @in N_BBH: number of BBHs
    @in random_pairing: if True, use uniform random pairing instead of mass-weighted (m^5)

    @out: all inputs
    """
    
    if k_3bb>0:

        # approx_mBH_sampling: 0 = exact (np.random.choice / rejection + np.where);
        # 1 = Fenwick-tree sampling. The 3bb draw is UNIFORM over the single BHs
        # (with rejection on the interaction term), so the Fenwick tree carries
        # unit weights; building it once lets us draw distinct indices directly
        # (no np.where) and DEFER the binary-pair deletions to one end-of-step
        # compaction. The catalyst m3 is NOT consumed, so only k1, k2 are removed.
        use_fenwick = (approx_mBH_sampling == 1) and len(mBH) >= 3
        if use_fenwick:
            tree = FenwickTree(np.ones(len(mBH)))
            consumed = []
            n_alive = len(mBH)
            # interaction term decreases with mass -> its max is at the smallest
            # masses; the fixed initial min only underestimates later mins (mBH
            # shrinks), so the rejection bound stays valid (acceptance <= 1).
            min_m = np.min(mBH)
            max_iv = (min_m + min_m)**(-0.5) * (min_m + min_m + min_m)**(-0.5)

        for i in range(k_3bb):

            # sample the 3bb members, getting indices k1, k2 (the binary pair)
            # into mBH; m3 is the catalyst (mass only, not consumed):
            if use_fenwick:
                if n_alive < 3:
                    break
                if random_pairing:
                    pair = tree.sample_indices(2, replace=False)
                    if pair.size < 2:
                        break
                    k1, k2 = int(pair[0]), int(pair[1])
                    m1, m2 = mBH[k1], mBH[k2]
                    m3 = mBH_avg
                else:
                    trip = tree.sample_indices(3, replace=False)
                    while trip.size >= 3 and np.random.rand() >= \
                            ((mBH[trip[0]] + mBH[trip[1]])**(-0.5)
                             * (mBH[trip[0]] + mBH[trip[1]] + mBH[trip[2]])**(-0.5)) / max_iv:
                        trip = tree.sample_indices(3, replace=False)
                    if trip.size < 3:
                        break
                    k1, k2 = int(trip[0]), int(trip[1])
                    m1, m2, m3 = mBH[trip[0]], mBH[trip[1]], mBH[trip[2]]
                N_3bb+=1 # update number of three-body-binaries
            else:
                if len(mBH) < 3:  # not enough single BHs left to form a 3bb, exit loop
                    break

                N_3bb+=1 # update number of three-body-binaries

                # sample masses that form the 3bb:
                if random_pairing:
                    m1, m2 = np.random.choice(mBH, size=2, replace=False)
                    m3 = mBH_avg
                else:
                    m1, m2, m3 = fast_sample_3bodyBinary(mBH, n_samples=1, approx_mBH_sampling=approx_mBH_sampling)

                # find index locations of the sampled BHs:
                k1 = np.squeeze(np.where(mBH==m1))+0
                k2 = np.squeeze(np.where(mBH==m2))+0

                k1 = int(np.atleast_1d(k1)[0])
                k2 = int(np.atleast_1d(k2)[0])

                if k1 == k2:
                    # find next available index for k2
                    candidates = np.where(mBH == m2)[0]
                    k2 = int(candidates[1]) if len(candidates) > 1 else None
                    if k2 is None:
                        continue  # skip this 3bb formation, can't find two distinct BHs

            # initial hardness of newly formed 3bb:
            eta = sample_hardness()

            # semimajor axis:
            sma = G_Newton * m1 * m2 / eta / m3 / vBH**2

            # eccentricity (thermal):
            eccen = np.sqrt(np.random.rand())

            # append binary:
            binaries = np.append(binaries, [[np.random.randint(0, 999999999), 3, sma, eccen, m1, m2, sBH[k1], sBH[k2], gBH[k1], gBH[k2], t, z, 0, hBH[k1], hBH[k2]]], axis=0)

            # update number of in-cluster BBHs:
            N_BBH+=1

            # delete single BHs (the binary pair; the catalyst m3 stays):
            if use_fenwick:
                tree.remove(k1)
                tree.remove(k2)
                consumed.append(k1)
                consumed.append(k2)
                n_alive -= 2
            else:
                mBH = np.delete(mBH, [k1, k2])
                sBH = np.delete(sBH, [k1, k2])
                gBH = np.delete(gBH, [k1, k2])
                hBH = np.delete(hBH, [k1, k2])

        # Fenwick path: apply the deferred binary-pair deletions in one pass.
        if use_fenwick and consumed:
            mBH = np.delete(mBH, consumed)
            sBH = np.delete(sBH, consumed)
            gBH = np.delete(gBH, consumed)
            hBH = np.delete(hBH, consumed)

    return t, z, k_3bb, mBH_avg, binaries, mBH, sBH, gBH, hBH, vBH, N_3bb, N_BBH

# End of file.
