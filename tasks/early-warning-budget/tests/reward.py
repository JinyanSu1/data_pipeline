"""Reward transform for the `early-warning-budget` task.

    if fpr > FPR_LIMIT:  reward = 0.0            # hard gate, no partial credit
    else:                reward = clamp(mean_lead_steps / L_MAX, 0.0, 1.0)

L_MAX is a GENERATOR CONSTANT, not a score anybody achieved.
=============================================================
It is the mean, over the 10 hidden test episodes that have a `t_obvious`, of

        t_obvious - onset

i.e. the number of timesteps between the earliest instant at which any
build-up signal exists in the data (the first sample on which the originating
group's factor loading is non-zero) and the instant at which the break becomes
obvious under the published trailing-20-step / rho > 0.7 rule.  It is therefore
an information bound: a detector with zero estimation delay -- physically
impossible, since correlation cannot be estimated from a single sample -- would
score exactly 1.0.  Nothing about any reference implementation enters it.

Derivation (reproduce with `python3 provenance/data_gen.py`, which prints it):

    test episode   onset   t_obvious   t_obvious - onset
    test-00         1046        1090          44
    test-01          934         990          56
    test-02          685         745          60
    test-03         1317        1348          31
    test-04          820         867          47
    test-05         1565        None          -- (never obvious, excluded)
    test-06         1707        1757          50
    test-07          267         306          39
    test-08          419         461          42
    test-09         1459        1496          37
    test-10          525        None          -- (never obvious, excluded)
    test-11         1198        1259          61
                                        ------
    mean over the 10 scored episodes =  46.7
"""

from __future__ import annotations

FPR_LIMIT = 0.05

# mean(t_obvious - onset) over the 10 scored hidden test episodes; see above.
L_MAX = 46.7


def compute_reward(mean_lead_steps: float, fpr: float) -> float:
    if fpr > FPR_LIMIT:
        return 0.0
    value = mean_lead_steps / L_MAX
    return float(min(1.0, max(0.0, value)))
