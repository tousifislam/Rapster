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
from .remnant import *
from .auxiliary import FenwickTree

def fast_sample_2capture(mBH, n_samples=1, weights=None, weight_sum=None, max_weight_factor=None):
    """
    Fast sampling for using rejection sampling with complexity O(N).

    @in mBH: array of BH masses
    @in n_samples: number of samples (m1, m2)
    @in weights, weight_sum, max_weight_factor: optional precomputed proposal.
        Default (None) -> rebuild the proposal from mBH each call (exact, the
        default path). Approx path: the caller passes the precomputed/
        incrementally-maintained m^(2/7) weights, their running sum, and the
        (fixed) max_weight_factor, avoiding the per-call O(N) pow/sum/max.
    """

    if weights is None:
        # Pre-calculate individual weights based on the separable part: m^(2/7)
        # This acts as our "proposal" distribution.
        weights = mBH**(2/7)
        weight_sum = weights.sum()
        # The maximum possible value of the non-separable part (m1 + m2)^(10/7)
        # used to normalize the rejection criteria.
        max_weight_factor = (2 * np.max(mBH))**(10/7)

    p_proposal = weights / weight_sum

    results = []
    while len(results) < n_samples:
        # Sample two candidates based on p_proposal
        candidates = np.random.choice(mBH, size=2, replace=False, p=p_proposal)
        m1_cand, m2_cand = candidates
        
        # Acceptance/Rejection step
        # We check the "interaction" term: (m1 + m2)^(10/7)
        acceptance_prob = (m1_cand + m2_cand)**(10/7) / max_weight_factor
        
        if np.random.rand() < acceptance_prob:
            results.append((m1_cand, m2_cand))
            
    return results[0] if n_samples == 1 else results

def two_body_capture(seed, t, dt, z, zCl_form, k_2cap, mBH_avg, binaries, mBH, sBH, gBH, hBH, vBH, v_star, N_2cap, N_BH, N_BBH, N_me, N_meRe, N_meEj, mergers, random_pairing=False, approx_mBH_sampling=False):
    """
    @in seed: simulation seed number
    @in t: simulation time
    @in dt: simulation time step
    @in z: simulation redshift
    @in zCl_form: cluster formation redshift
    @in k_2cap: number of 2-captures in current step
    @in mBH_avg: average BH mass
    @in binaries: array of BBHs
    @in mBH: array of single BH masses
    @in sBH: array of single BH spins
    @in gBH: array of single BH generations
    @in hBH: array of BH tdes count
    @in vBH: 3D BH velocity dispersion
    @in v_star: 3D star velocity dispersion
    @in N_2cap: number of 2-captures
    @in N_BH: number of BHs
    @in N_BBH: number of BBHs
    @in N_me: number of mergers
    @in N_meRe: number of retained mergers
    @in N_meEj: number of ejected mergers
    @in mergers: array of mergers: [seed, ind, channel, a, e, m1, m2, s1, s2, g1, g2, theta1, theta2, dPhi, t_form, z_form, t_merge, z_merge, m_rem, s_rem, g_rem, vGW_kick, s_eff, q, v_esc, h1, h2]
    @in random_pairing: if True, use uniform random pairing instead of mass-weighted (m^2)

    @out: all inputs
    """
    
    if k_2cap > 0:

        mBH_temp = []
        sBH_temp = []
        gBH_temp = []
        hBH_temp = []

        # approx_mBH_sampling: 0 = exact (np.random.choice / fast_sample_2capture);
        # 1 = Fenwick-tree sampling. The Fenwick path builds the m^(2/7) proposal
        # tree ONCE, draws + removes BHs in O(log N) (the tree returns indices, so
        # no np.where), and DEFERS the BH-array deletions to a single end-of-step
        # compaction (consumed indices collected below). Statistically equivalent
        # to the exact path (validated by distribution), not bit-identical.
        use_fenwick = (approx_mBH_sampling == 1) and len(mBH) >= 2
        if use_fenwick:
            fw_weights = np.ones(len(mBH)) if random_pairing else mBH**(2/7)
            tree = FenwickTree(fw_weights)
            consumed = []
            n_alive = len(mBH)
            max_wf = (2 * np.max(mBH))**(10/7)   # fixed rejection bound (mBH only shrinks)

        for i in range(k_2cap):

            # sample the two BHs that form the captured binary, getting their
            # indices k1, k2 into the (current) mBH array:
            if use_fenwick:
                if n_alive < 2:
                    break
                pair = tree.sample_indices(2, replace=False)
                if not random_pairing:
                    # rejection on the interaction term (same target as fast_sample_2capture)
                    while pair.size >= 2 and np.random.rand() >= (mBH[pair[0]] + mBH[pair[1]])**(10/7) / max_wf:
                        pair = tree.sample_indices(2, replace=False)
                if pair.size < 2:
                    break
                k1, k2 = int(pair[0]), int(pair[1])
                m1, m2 = mBH[k1], mBH[k2]
            else:
                if len(mBH) < 2:
                    break # avoid infinite loop in the fast_sample_2capture function.

                if random_pairing:
                    m1, m2 = np.random.choice(mBH, size=2, replace=False)
                else:
                    m1, m2 = fast_sample_2capture(mBH, n_samples=1)

                # find index locations of the sampled BHs:
                k1 = np.squeeze(np.where(mBH==m1))+0
                k2 = np.squeeze(np.where(mBH==m2))+0

                k1 = int(np.atleast_1d(k1)[0])
                k2 = int(np.atleast_1d(k2)[0])

                if k1 == k2:
                    candidates = np.where(mBH == m2)[0]
                    k2 = int(candidates[1]) if len(candidates) > 1 else None
                    if k2 is None:
                        continue

            s1 = sBH[k1]; g1 = gBH[k1]; h1 = hBH[k1]
            s2 = sBH[k2]; g2 = gBH[k2]; h2 = hBH[k2]
            
            ind = np.random.randint(0, 999999999)
            
            theta1, theta2, dPhi = sample_angles()
            
            m_rem, s_rem, vGW_kick = merger_remnant(m1, m2, sBH[k1], sBH[k2], theta1, theta2, dPhi)
            g_rem = np.max([gBH[k1], gBH[k2]]) + 1
            h_rem = h1 + h2

            # relative velocity:
            v_rel = get_maxwell_sample(np.sqrt(2/3) * vBH)
            
            # total mass:
            m12 = m1 + m2
            
            # reduced mass:
            mu = m1*m2/m12
            
            # maximum impact parameter for capture:
            b_max = (340 * np.pi / 3)**(1/7) * m12**(6/7) * mu**(1/7) / v_rel**(9/7) * G_Newton * c_light**(-5/7)

            # impact parameter sampled from uniform in b^2 distribution:
            b = np.sqrt(np.random.rand() * b_max**2)
            
            # pericenter distance:
            rp = b**2 * v_rel**2 / 2 / G_Newton / m12
            
            # GW energy released:
            E_gw = 85 * np.pi / 12 / np.sqrt(2) * mu**2 * m12**(5/2) / rp**(7/2) * G_Newton**(7/2) / c_light**5
            
            # final energy:
            E_fin = mu * v_rel**2 / 2 - E_gw
            
            max_iter = 1000  # safety cap to prevent infinite loop for extreme mass/velocity combinations
            n_iter = 0
            # make sure eccentricity is strictly smaller than unity:
            while 1 + 2 * E_fin * b**2 * v_rel**2 / m12**2 / mu / G_Newton**2 < 0:
                n_iter += 1
                if n_iter > max_iter:  # could not find valid eccentricity, skip this capture event
                    break
                
                # impact parameter sampled from uniform in b^2 distribution:
                b = np.sqrt(np.random.rand() * b_max**2)
                
                # pericenter distance:
                rp = b**2 * v_rel**2 / 2 / G_Newton / m12
                
                # GW energy released:
                E_gw = 85 * np.pi / 12 / np.sqrt(2) * mu**2 * m12**(5/2) / rp**(7/2) * G_Newton**(7/2) / c_light**5
                
                # final energy:
                E_fin = mu * v_rel**2 / 2 - E_gw
                
            if n_iter > max_iter:  # capture event skipped due to no valid eccentricity found
                continue
                
            # semimajor axis at formation:
            sma = - G_Newton * m12 * mu / 2 / E_fin
            
            # eccentricity at formation:
            eccen = np.sqrt(1 + 2 * E_fin * b**2 * v_rel**2 / m12**2 / mu / G_Newton**2)
            
            # consume the two captured BHs. Fenwick: mark them removed in the tree
            # (O(log N)) and record the indices for a single end-of-step compaction.
            # Exact: delete from the BH arrays now (indices track the shrinking mBH).
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

            N_2cap+=1
            
            # check if binary merges within the current step:
            if T_GW(m1, m2, sma, eccen) < np.min([dt, lookback_interp(zCl_form) - t]):
                
                if vGW_kick < 2 * np.sqrt(v_star**2 + vBH**2): # merger remnant retained in cluster
                    
                    mBH_temp.append(m_rem)
                    sBH_temp.append(s_rem)
                    gBH_temp.append(g_rem)
                    hBH_temp.append(h_rem)
                    
                    N_BH = N_BH - 1
                    
                    N_meRe+=1
                        
                else: # merger remnant ejected from cluster
                        
                    N_BH = N_BH - 2
                        
                    N_meEj+=1
                        
                N_me+=1
                    
                # order BHs by mass:
                mA = m1; sA = s1; gA = g1; hA = h1; thetaA = theta1
                mB = m2; sB = s2; gB = g2; hB = h2; thetaB = theta2
                if mA>mB:
                    m1 = mA; s1 = sA; g1 = gA; h1 = hA; theta1 = thetaA
                    m2 = mB; s2 = sB; g2 = gB; h2 = hB; theta2 = thetaB
                else:
                    m1 = mB; s1 = sB; g1 = gB; h1 = hB; theta1 = thetaB
                    m2 = mA; s2 = sA; g2 = gA; h2 = hA; theta2 = thetaA
                    
                # mass ratio:
                q = m2 / m1
                
                # effective spin parameter:
                s_eff = (m1 * s1 * np.cos(theta1) + m2 * s2 * np.cos(theta2)) / (m1 + m2)
                
                # append merger:
                mergers.append([seed, ind, 2, sma, eccen, m1, m2, s1, s2, g1, g2, theta1, theta2, dPhi, t, z, t + T_GW(m1, m2, sma, eccen),
                                redshift_interp(lookback_interp(zCl_form) - t - T_GW(m1, m2, sma, eccen)), m_rem, s_rem, g_rem, vGW_kick, s_eff, q, 2*v_star, h1, h2])

            else:
                
                # append binary:
                binaries = np.append(binaries, [[ind, 2, sma, eccen, m1, m2, s1, s2, g1, g2, t, z, 0, h1, h2]], axis=0)
                
                N_BBH+=1
                
        # Fenwick path: apply the deferred BH-array deletions in a single pass
        # (consumed holds the indices into the unmodified mBH/sBH/gBH/hBH).
        if use_fenwick and consumed:
            mBH = np.delete(mBH, consumed)
            sBH = np.delete(sBH, consumed)
            gBH = np.delete(gBH, consumed)
            hBH = np.delete(hBH, consumed)

        mBH_temp = np.array(mBH_temp)
        sBH_temp = np.array(sBH_temp)
        gBH_temp = np.array(gBH_temp)
        hBH_temp = np.array(hBH_temp)

        mBH = np.append(mBH, mBH_temp)
        sBH = np.append(sBH, sBH_temp)
        gBH = np.append(gBH, gBH_temp)
        hBH = np.append(hBH, hBH_temp)
        
    return seed, t, dt, z, zCl_form, k_2cap, mBH_avg, binaries, mBH, sBH, gBH, hBH, vBH, v_star, N_2cap, N_BH, N_BBH, N_me, N_meRe, N_meEj, mergers

# End of file.
