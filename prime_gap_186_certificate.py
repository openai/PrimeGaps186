#!/usr/bin/env python3
"""Recompute the complete numerical certificate for the gap-186 trial.

Requires Python 3.10+, NumPy, and Python-FLINT linked to a FLINT build with
corrected signed FFT integer-polynomial convolution. A mandatory startup
regression detects the known defect; this script does not modify dependencies.

All trial, geometry, bin and Young parameters are embedded exactly. No saved
integrals, certificate results or project modules are read.
"""

import argparse
import ctypes
import gc
import json
import math
import multiprocessing as mp
import sys
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
from math import comb, prod
from pathlib import Path
from time import monotonic
from types import SimpleNamespace

import numpy as np
from flint import arb, arb_poly, arb_series, ctx, fmpz_poly

# Exact trial, source ladders, cell masks and component schedule.
# Matrix rows follow these signatures; columns are radial degrees 0 through 6.
# fmt: off
COEFFICIENT_SIGNATURES = [[], [2], [3], [4], [5], [6], [2, 2], [2, 3], [2, 4], [3, 3], [2, 2, 2]]


COEFFICIENT_INTEGER_MATRIX = [
    [10000000000, -264598476112, 834262268474, -3540575351215, 5377491111325, 116705356572254,
     121730820431102],
    [71070047507, 7222861788586, -48747932657986, 290976672545723, -1136422724027134,
     -2058910631434711, 1375878942948547],
    [6457252424873, -61201446212885, 28811649792090, -2058803084231281, 9970156747759406,
     38278849934023144, 34806920812932737],
    [-16779263512274, 128033707910825, 169290603857215, 7359669931312727, -35090795379920588,
     -120997133235510923, -141527585901304670],
    [28114161526671, -276375633435566, -690482702304933, -15120456749385986, 47650276584638031,
     372137534144492224, 2191913505882230103],
    [-21150553032771, 379875924448266, -929062247569514, -13967219843969236, 168416050564605519,
     -248058724714138769, -4161797957833172083],
    [-19004216224617, 136674974139652, -183344485344333, 1114879023072977, 1119583327002910,
     -47568062965320963, 2537631525616777],
    [31171802814567, -97428388152051, 526635309233054, 1222047580792220, 15846796161434448,
     -165623953461271580, -2358019996221938729],
    [-53602626608739, 329397902441121, 1613505134333316, 13893066541430270, -126262434123562668,
     239934302929501943, -1506861482386002243],
    [27653330903418, -290549334488305, -1847330641348475, 28351866831729468, -61505472221886320,
     -424266003419714347, 2585236507449911535],
    [12374547113901, -244168600145684, 1603694896120437, -21603130476787649, -30285492734943698,
     381987976419637874, 3605516061450295448],
]


YOUNG_Q = [
    [961904, 502424, 483341, 547373, 563915, 583181, 604629, 620671, 629321, 635211, 616326,
     593862, 573977, 553178, 531463, 508862, 459016],
    [7266522, 1241454497, 1208324400, 1152630107, 1126190783, 1096246679, 1058983690, 967816560,
     867471653, 603785822, 32188902, 1308239, 386321, 373849, 377891, 385136, 395013, 405835,
     419505, 432001, 445139, 457321, 525975, 518733, 515168, 512357, 509770, 507320, 504951,
     502604, 503256, 498048, 492222, 485810, 433769],
]
# fmt: on


POLICY = {
    "intervals": 98304,
    "convolution_length": 98264,
    "arb_precision_bits": 160,
    "cap_fractional_bits": 224,
    "source_fractional_bits": 192,
    "production_cap_threads": 1,
    "production_source_threads": 2,
    "factorial_degree": 3,
    "cap_dickman_method": "trapezoid",
    "exact_eulerian_count_limit": 32,
    "low_dickman_floor": str(F(1, 10**40)),
    "low_dickman_method": "trapezoid_with_renewal_fallback",
    "low_trapezoid_minimum_cap_cells": 64,
    "low_subdivision_refinement": 1,
    "monotone_dickman_extension": False,
    "exact_per_bin_core_tightening": True,
    "low_radial_factorization": True,
    "low_inner_dual_jets": False,
    "rank_owner_feasible_palm_mass": True,
    "rank_same_owner_feasible_palm_mass": True,
    "positive_polynomial_upper_rounding": "nonnegative_dense_shift_then_floor",
    "signed_reduction": "directed_binary64_53_bits_ties_to_even_gradual_underflow_no_overflow",
    "frozen_young_parameters": True,
    "cap_decimal_scale": 10**24,
    "raw_relative_decimal_scale": 10**18,
    "component_relative_decimal_scale": 10**12,
    "young_denominator": 10**6,
    "source_normalization_denominator": "floor(fresh_cap_I_lower*cap_decimal_scale)/cap_decimal_scale",
}


def _ladder(name, T, eps, limit, cs, S, rho, gap):
    """Build an exact source ladder, increasing the order every twelve steps."""
    previous = F(0)
    E = rho * (S + T) - F(1, 2)
    rows = []
    for index in range(100):
        order = min(index // 12 + 1, 3)
        c, slope = cs[order - 1]
        omega = min(limit, (c - eps - E + 2 * previous - gap) / slope)
        delta = c - slope * omega - eps
        B = (F(1, 2) + 2 * previous) / rho
        xi, a, b = delta / rho, B - T, B - S
        eta = xi if order < 3 else (xi + S + T - B) / 2
        A, C = a + eta, b + eta
        assert previous < omega <= limit and delta > 0
        rows.append(
            dict(
                ladder=name,
                index=index,
                source_order=order,
                previous=previous,
                omega=omega,
                delta=delta,
                B=B,
                upper_B=(F(1, 2) + 2 * omega) / rho,
                xi=xi,
                a=a,
                b=b,
                A=A,
                C=C,
                eta_D=eta,
                eta_E=eta,
                owner_model="linear" if order < 3 else "capped_outer_linear",
                owner_plateau=None if order < 3 else 23 * C / 40,
            )
        )
        if omega == limit:
            return rows
        previous = omega
    raise ArithmeticError("source ladder did not reach its prescribed limit")


def _shells(pairs):
    """Describe consecutive radial shells, starting at zero."""
    lower, result = F(0), []
    for upper, cap in pairs:
        result.append(dict(lower=str(lower), upper=str(upper), ceiling=str(cap)))
        lower = upper
    return result


def _cells(shells, dimension, n, h):
    """Select grid cells for each shell and round its ceiling down."""
    result = []
    for index, row in enumerate(shells):
        lower, upper, cap = (F(row[k]) for k in ("lower", "upper", "ceiling"))
        first = max(0, lower // h - dimension + 1)
        last = min(n - 1, upper // h - dimension)
        if first <= last:
            result.append(
                dict(
                    shell_index=index,
                    first_index=first,
                    last_index=last,
                    ceiling=str((cap // h) * h),
                    assigned_radial_lower=str(lower),
                    assigned_radial_upper=str(upper),
                    physical_total_lower=str(first * h),
                    physical_total_upper=str((last + dimension) * h),
                )
            )
    return result


def _event_cells(cells, dimension, h, lower, upper):
    """Restrict a cell cover to the event's radial interval."""
    result = []
    for row in cells:
        first = max(row["first_index"], lower // h - dimension + 1)
        last = min(row["last_index"], upper // h)
        if first <= last:
            result.append(
                dict(
                    row,
                    first_index=first,
                    last_index=last,
                    physical_total_lower=str(first * h),
                    physical_total_upper=str((last + dimension) * h),
                )
            )
    return result


def _group(identifier, role, dimension, order, rows, orientation, lower, upper, cells, h):
    """Combine source rows and record the guards for their three source bounds."""
    threshold_key, core_key = ("A", "a") if orientation == "outer" else ("C", "b")
    ceiling = min(row[threshold_key] for row in rows)
    activation = min(row["xi"] for row in rows)
    pieces = _event_cells(cells, dimension, h, lower, upper)
    hard_cap = max(F(row["ceiling"]) for row in pieces)
    low_index = activation // h
    maximum_split_index = min(hard_cap // h, (ceiling / order) // h)
    split_index = min(max(low_index + 1, (ceiling / (2 * order)) // h), maximum_split_index)
    split = split_index * h
    assert 2 <= low_index < split_index <= maximum_split_index
    rank_lower = max(activation, split, ceiling / (order + 1))
    second_lower = max(activation, split, (ceiling - hard_cap) / order)
    return dict(
        id=identifier,
        role=role,
        dimension=dimension,
        order=str(order),
        ceiling=str(ceiling),
        activation=str(activation),
        radial_lower=str(lower),
        radial_upper=str(upper),
        hard_cap=str(hard_cap),
        split=str(split),
        nonlargest=True,
        empty=False,
        aligned_cap_pieces=pieces,
        source_rows=[
            dict(
                ladder=row["ladder"],
                index=row["index"],
                source_order=row["source_order"],
                core=str(row[core_key]),
                activation=str(row["xi"]),
                ceiling=str(row[threshold_key]),
                owner_model=row["owner_model"],
                owner_plateau=None if row["owner_plateau"] is None else str(row["owner_plateau"]),
            )
            for row in rows
        ],
        low_bin_guard=dict(
            first_requested_endpoint=str(activation),
            outward_first_endpoint=str(low_index * h),
            first_grid_index=low_index,
            final_endpoint=str(split),
            final_grid_index=split_index,
            maximum_rank_one_safe_endpoint=str(maximum_split_index * h),
            rank_one_margin=str(ceiling - order * split),
            hard_cap_margin=str(hard_cap - split),
            convention="each requested bin is enlarged to floor(low/h)h,ceil(high/h)h",
        ),
        rank_two_guard=dict(
            largest_lower=str(rank_lower),
            largest_upper=str(hard_cap),
            second_lower=str(second_lower),
            positive_remaining_threshold=str(ceiling - hard_cap),
            second_cutoff="max(split,activation,(ceiling-q)/order)",
            may_overcover_by_omitting_split_and_activation=True,
            old_largest_cap_guard_not_assumed=True,
        ),
    )


def _schedule(groups, hybrid):
    """Partition each source group into low, rank-two and high components."""
    a = [F(1, 20) * F(6, 5) ** j for j in range(10)]
    x = [F(g["activation"]) for g in groups]
    p = [F(g["split"]) for g in groups]
    low_sequences = (
        (x[0], 3 * x[0] / 2, a[0], a[4], a[5], a[6], a[7], a[8], (a[8] + a[9]) / 2, a[9], p[0]),
        tuple(x[1] * 2**j for j in range(9))
        + (F(1, 100), F(3, 200), F(9, 400), F(27, 800))
        + tuple(a[:8])
        + ((a[7] + p[1]) / 2, p[1]),
        (x[2], a[0], a[4], a[6], a[8], p[2]),
        (x[3], 2 * x[3], F(1, 100), F(27, 800), a[3], a[5], a[6], p[3]),
        (x[4], a[1], a[4], a[5], a[7], a[8], p[4]),
        (
            x[5],
            2 * x[5],
            16 * x[5],
            64 * x[5],
            256 * x[5],
            F(9, 400),
            a[1],
            a[3],
            a[5],
            a[6],
            a[7],
            p[5],
        ),
    )
    rank_fractions = (
        tuple(F(j, 6) for j in range(7)),
        tuple(F(j, 16) for j in range(9)) + (F(5, 8), F(3, 4), F(7, 8), F(1)),
        (F(0), F(1)),
        (F(0), F(1, 2), F(1)),
        (F(0), F(1, 6), F(1, 2), F(2, 3), F(1)),
        (F(0), F(1, 8), F(3, 8), F(1, 2), F(3, 4), F(1)),
    )
    tasks = []
    for gi, (group, low, rank) in enumerate(zip(groups, low_sequences, rank_fractions)):
        local = []
        assert low[0] == x[gi] and low[-1] == p[gi] and all(u < v for u, v in zip(low, low[1:]))
        for j, (lower, upper) in enumerate(zip(low, low[1:])):
            slope = 120 if gi == 0 and j == 2 else math.ceil(F(9 if gi == 5 else 7) / upper)
            local.append(
                dict(
                    group=group["id"],
                    kind="low",
                    index=j,
                    label=f"L_{{{j + 1}}}",
                    parameters=dict(low=str(lower), high=str(upper), slope=str(slope)),
                )
            )
        q0 = F(group["ceiling"]) / (F(group["order"]) + 1)
        z = F(group["hard_cap"])
        for j, (lower, upper) in enumerate(zip(rank, rank[1:])):
            local.append(
                dict(
                    group=group["id"],
                    kind="rank_two",
                    index=j,
                    label=f"P_{{{j + 1}}}",
                    parameters=dict(
                        q_low=str(q0 + lower * (z - q0)), q_high=str(q0 + upper * (z - q0))
                    ),
                )
            )
        local.append(dict(group=group["id"], kind="high", index=0, label="H", parameters={}))
        for j, task in enumerate(local):
            task["source_group_id"] = group["id"]
            if gi < 2:
                q = YOUNG_Q[gi][j]
                task.update(young_q=q, young_denominator=10**6, young=str(F(q, 10**6)))
            else:
                task["restoration_coefficient"] = hybrid["d0" if gi < 4 else "one_minus_b"]
        tasks.extend(local)
    return tasks


def build_inputs():
    """Derive the complete trial, cell cover and source schedule from exact inputs."""
    k, N = 40, 98304
    gap, tau = F(1, 10**7), F(1, 10**10)
    rho = F(1, 4) + F(12499, 10**6)
    rs = rho - gap
    S, T1 = F(2742997, 10**7) / rs, F(251, 1000) / rs
    T0 = 2 - F(3, 1000) - S
    h, e, zeta = S / N, gap / rho, F(19037, 100000) / rho
    sigma0, sigmam = F(100001, 10**6), F(1, 2) - F(40481, 100000) + tau
    old_cs = (((1 - 5 * sigma0) / 15, F(18, 5)), ((1 - 4 * sigma0) / 16, F(7, 2)), (F(3, 80), F(3)))
    new_cs = (
        ((1 - 5 * sigmam) / 15, F(18, 5)),
        ((1 - 4 * sigmam) / 16, F(7, 2)),
        ((1 - 2 * sigmam) / 20, F(16, 5)),
    )
    old_all = _ladder("old", T0, F(1, 10**6), F(12499, 10**6), old_cs, S, rho, gap)
    new_all = _ladder("new", T1, F(1, 10**7), F(253, 20000), new_cs, S, rho, gap)
    shells = dict(
        outer=_shells(
            (
                (new_all[0]["a"], zeta),
                (new_all[24]["a"], (S + e) / 2),
                (S, S + e / 2 - 23 * (T1 + e / 2) / 40),
            )
        ),
        base=_shells(
            (
                (old_all[12]["b"], zeta),
                (old_all[24]["b"], (T0 + e) / 2),
                (T0, 63 * (T0 + e / 2) / 160),
            )
        ),
        enlarged=_shells(
            (
                (new_all[12]["b"], zeta),
                (new_all[24]["b"], (T1 + e) / 2),
                (T1, 63 * (T1 + e / 2) / 160),
            )
        ),
        full=_shells(((S, zeta),)),
    )
    cells = {
        name: _cells(layers, k if name == "outer" else k - 1, N - k, h)
        for name, layers in shells.items()
    }
    maxima = {
        name: max(F(row["physical_total_upper"]) for row in rows) for name, rows in cells.items()
    }
    old = [r for r in old_all if r["B"] < maxima["outer"] + maxima["base"]]
    new = [r for r in new_all if r["B"] < maxima["outer"] + maxima["enlarged"]]
    assert (len(old_all), len(new_all), len(old), len(new)) == (29, 43, 28, 39)
    groups = []
    for role, rows, orientation, cell_key in (
        ("outer", old + new, "outer", "outer"),
        ("old_inner", old, "inner", "base"),
        ("new_inner", new, "inner", "enlarged"),
    ):
        h2 = [
            r
            for r in rows
            if r["source_order"] < 3 and (orientation == "outer" or r["source_order"] == 2)
        ]
        h25 = [r for r in rows if r["source_order"] == 3]
        core = "a" if orientation == "outer" else "b"
        c2, c25 = min(r[core] for r in h2), min(r[core] for r in h25)
        d = k if orientation == "outer" else k - 1
        groups.append(
            _group(role + "_h2", role, d, F(2), h2, orientation, c2, c25, cells[cell_key], h)
        )
        groups.append(
            _group(
                role + "_h25",
                role,
                d,
                F(5, 2),
                h25,
                orientation,
                c25,
                maxima[cell_key],
                cells[cell_key],
                h,
            )
        )
    mass, K, lam = F(49999, 50000), F(17, 50), F(1, 125)
    ah, bh = mass * mass - mass * lam, (1 - mass / lam) * (1 - mass) * K
    hybrid = dict(
        mass=str(mass),
        pair_constant=str(K),
        **{"lambda": str(lam)},
        a=str(ah),
        b=str(bh),
        d0=str(1 - ah - bh),
        one_minus_b=str(1 - bh),
        outer_absolute_weights=dict(
            base="1",
            enlarged_minus_base=str(max(abs(ah + bh), abs(bh))),
            outside_enlarged=str(abs(bh)),
        ),
    )
    parameters = dict(
        S=str(S),
        T0=str(T0),
        T1=str(T1),
        rho=str(rho),
        rho_star=str(rs),
        normalized_global_cap=str(zeta),
        fixed_tau=str(tau),
        physical_outer=str(rs * S),
        physical_inner_base=str(rs * T0),
        physical_inner_new=str(rs * T1),
    )
    manifest = dict(
        parameters=parameters,
        common_outer_shells=shells["outer"],
        base_inner_shells=shells["base"],
        enlarged_inner_shells=shells["enlarged"],
        full_inner_shells=shells["full"],
    )
    trial = dict(
        dimension=k,
        center="9/10",
        coefficients=[str(F(c, 10**10)) for row in COEFFICIENT_INTEGER_MATRIX for c in row],
        descriptors=[
            [list(signature), degree] for signature in COEFFICIENT_SIGNATURES for degree in range(7)
        ],
        profile=dict(weight="21/200", slow="1/100", fast="907/5"),
        hybrid={key: hybrid[key] for key in ("mass", "pair_constant", "lambda")},
        source_geometry=manifest,
        common_self_source_parameters=dict(
            omega="31/10000", delta="21319/800000", new_mixed_witness_index=12
        ),
    )
    tasks = _schedule(groups, hybrid)
    return dict(
        trial=trial,
        source_groups=groups,
        tasks=tasks,
        hybrid=hybrid,
        source_ladders=dict(old=old_all, new=new_all),
        cells=cells,
        shells=shells,
        layout=dict(
            intervals=N,
            convolution_length=N - k,
            grid_step=str(h),
            maxima={key: str(value) for key, value in maxima.items()},
        ),
    )


def _json_exact(value):
    """Keep exact fractions as rational strings in the JSON-ready input tree."""
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_exact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_exact(item) for item in value]
    return value


DERIVED_INPUTS = _json_exact(build_inputs())
TRIAL = DERIVED_INPUTS["trial"]
SOURCE_GROUPS = DERIVED_INPUTS["source_groups"]
TASKS = DERIVED_INPUTS["tasks"]
GROUP_BY_ID = {group["id"]: group for group in SOURCE_GROUPS}
SOURCE_LADDERS = DERIVED_INPUTS["source_ladders"]
LAYOUT = DERIVED_INPUTS["layout"]
CAP_SIGNATURES = tuple(map(tuple, COEFFICIENT_SIGNATURES))
CAP_COEFFICIENTS = tuple(map(tuple, COEFFICIENT_INTEGER_MATRIX))
CAP_SHELL_DATA = {
    name: tuple((s["upper"], s["ceiling"]) for s in shells)
    for name, shells in DERIVED_INPUTS["shells"].items()
}


# Directed arithmetic, Dickman bounds and cap forms.
def check_flint_signed_fft():
    """Reject the known signed-FFT regression; never modify the library."""
    a, b = (1 << 509) - 1, (1 << 510) - 1
    p, q = fmpz_poly([a] * 16), fmpz_poly([-b] * 16)
    expected = [-min(j + 1, 31 - j, 16) * a * b for j in range(31)]
    if p * q != fmpz_poly(expected) or p.mul_low(q, 16) != fmpz_poly(expected[:16]):
        raise RuntimeError(
            "FLINT failed its signed-FFT regression; install a build with corrected signed convolution."
        )


LABELS = ("base", "enlarged", "full")


def rational(value):
    value = F(value)
    return arb(value.numerator) / value.denominator


def nonnegative(value, upper_bound=None):
    low, high = value.lower(), value.upper()
    if not value.is_finite() or high < 0:
        raise ArithmeticError("invalid positive enclosure")
    low = max(arb(0), low)
    if upper_bound is not None:
        high = min(high, arb(upper_bound))
    if low > high:
        raise ArithmeticError("empty positive enclosure")
    return low.union(high).nonnegative_part()


@dataclass(frozen=True)
class DickmanGrid:
    step: F
    cap_index: int
    requested_cap: F
    lower: list
    upper: list

    @property
    def cap(self):
        return self.step * self.cap_index

    def endpoint_interval(self, j):
        assert 0 <= j < len(self.lower)
        return self.lower[j].union(self.upper[j])

    def conditional_cell_masses(self):
        """Cell masses with one shared step and trapezoid error."""
        h, c = rational(self.step), rational(self.cap)
        error = h**3 / (12 * c**2)
        for j in range(len(self.lower) - 1):
            if j < self.cap_index:
                yield h, h
                continue
            low = (h * (self.lower[j] + self.lower[j + 1]) / 2 - error).lower()
            high = (h * (self.upper[j] + self.upper[j + 1]) / 2).upper()
            low = max(arb(0), low)
            if low > high:
                raise ArithmeticError("invalid Dickman cell mass")
            yield low, high


def outward_dickman_grid(*, cap, step, count, method="trapezoid"):
    cap, step, count = F(cap), F(step), int(count)
    if cap <= 0 or step <= 0 or count < 1 or method not in ("trapezoid", "riemann"):
        raise ValueError("invalid Dickman grid")
    M = int(cap / step)
    if M < 1:
        raise ValueError("cap contains no positive cell")
    lo, hi = [arb(1)] * (min(count, M) + 1), [arb(1)] * (min(count, M) + 1)
    h, c = rational(step), rational(step * M)
    error = h**3 / (2 * c**3)
    for j in range(M, count):
        if method == "riemann":
            mass = (arb(j + 1) / j).log()
            low = (lo[j] - hi[j - M] * mass).lower()
            high = (hi[j] - lo[j + 1 - M] * mass).upper()
        else:
            high_integrand = hi[j - M] / (2 * j) + hi[j + 1 - M] / (2 * (j + 1))
            low_integrand = lo[j - M] / (2 * j) + lo[j + 1 - M] / (2 * (j + 1))
            low = (lo[j] - high_integrand).lower()
            high = (hi[j] - low_integrand + error).upper()
        low, high = max(arb(0), low), min(arb(1), high)
        if not 0 <= low <= high <= 1 or low > lo[j] or high > hi[j]:
            raise ArithmeticError("Dickman grid lost order or monotonicity")
        lo.append(low)
        hi.append(high)
    return DickmanGrid(step, M, cap, lo, hi)


def verify_exact_first_delay(grid):
    last = min(2 * grid.cap_index, len(grid.lower) - 1)
    for j in range(grid.cap_index, last + 1):
        exact = 1 - (arb(j) / grid.cap_index).log()
        if not grid.endpoint_interval(j).overlaps(exact):
            raise ArithmeticError("Dickman first-delay inconsistency")
    return last - grid.cap_index + 1


def _scaled_dyadic(endpoint, bits, *, upper):
    if not endpoint.is_exact() or not endpoint.is_finite():
        raise ArithmeticError("expected an exact finite Arb endpoint")
    m, e = map(int, endpoint.man_exp())
    shift = e + bits
    if shift >= 0:
        return m << shift
    d = 1 << -shift
    return -((-m) // d) if upper else m // d


def _outward_scaled_positive_integer(value, bits, *, upper):
    value = int(value)
    if value < 0:
        raise ValueError("negative positive coefficient")
    if not value:
        return 0.0
    shift = max(0, value.bit_length() - 1000)
    if shift:
        value = ((value + (1 << shift) - 1) >> shift) if upper else value >> shift
    nearest = math.ldexp(float(value), shift - int(bits))
    result = float(np.nextafter(nearest, np.inf if upper else -np.inf))
    if not math.isfinite(result):
        raise ArithmeticError("nonfinite scaled coefficient")
    return max(0.0, result)


@lru_cache(maxsize=24)
def _coefficientwise_ceiling_shift(bits, length):
    if bits < 1 or length < 1:
        raise ValueError("positive fixed scale and polynomial length required")
    return fmpz_poly([(1 << bits) - 1] * length)


@dataclass(frozen=True, slots=True)
class PositiveFixedPolynomial:
    lower: fmpz_poly
    upper: fmpz_poly
    bits: int
    length: int

    def __post_init__(self):
        if self.bits < 32 or self.length < 1 or max(len(self.lower), len(self.upper)) > self.length:
            raise ValueError("invalid fixed polynomial size")

    @classmethod
    def from_arb(cls, values, bits, length=None):
        values = tuple(map(arb, values))
        length = len(values) if length is None else int(length)
        low, high = [], []
        if len(values) > length or bits < 32 or length < 1:
            raise ValueError("invalid fixed polynomial size")
        for value in values:
            if value.upper() < 0:
                raise ArithmeticError("negative positive input")
            low.append(max(0, _scaled_dyadic(value.lower(), bits, upper=False)))
            high.append(max(0, _scaled_dyadic(value.upper(), bits, upper=True)))
            if low[-1] > high[-1]:
                raise ArithmeticError("reversed fixed endpoints")
        return cls(fmpz_poly(low), fmpz_poly(high), bits, length)

    @classmethod
    def one(cls, bits, length):
        return cls(fmpz_poly([1 << bits]), fmpz_poly([1 << bits]), bits, length)

    @classmethod
    def zero(cls, bits, length):
        return cls(fmpz_poly([]), fmpz_poly([]), bits, length)

    def _compatible(self, other):
        if (self.bits, self.length) != (other.bits, other.length):
            raise ValueError("incompatible fixed polynomials")

    def multiply(self, other):
        self._compatible(other)
        low = self.lower.mul_low(other.lower, self.length) // (1 << self.bits)
        p = self.upper.mul_low(other.upper, self.length)
        high = (
            (p + _coefficientwise_ceiling_shift(self.bits, len(p))) // (1 << self.bits)
            if len(p)
            else fmpz_poly([])
        )
        return type(self)(low, high, self.bits, self.length)

    def add(self, other):
        self._compatible(other)
        return type(self)(
            self.lower + other.lower, self.upper + other.upper, self.bits, self.length
        )

    def scale_integer(self, factor):
        if factor < 0 or int(factor) != factor:
            raise ValueError("negative or noninteger positive scalar")
        return type(self)(self.lower * factor, self.upper * factor, self.bits, self.length)

    def binary64_intervals(self):
        rows = []
        for upper, polynomial in ((False, self.lower), (True, self.upper)):
            row = np.zeros(self.length)
            row[: len(polynomial)] = [
                _outward_scaled_positive_integer(v, self.bits, upper=upper)
                for v in polynomial.coeffs()
            ]
            rows.append(row)
        return _cap_checked_interval(tuple(rows))


def _cap_checked_interval(values):
    low, high = values
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low > high):
        raise ArithmeticError("invalid binary64 interval")
    return values


def float_interval(values):
    low = np.fromiter((float(v.lower()) for v in values), float, count=len(values))
    high = np.fromiter((float(v.upper()) for v in values), float, count=len(values))
    return _cap_checked_interval((np.nextafter(low, -np.inf), np.nextafter(high, np.inf)))


def interval_multiply(a, b):
    p = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return _cap_checked_interval(
        (np.nextafter(np.minimum.reduce(p), -np.inf), np.nextafter(np.maximum.reduce(p), np.inf))
    )


def _interval_add(a, b):
    return _cap_checked_interval(
        (np.nextafter(a[0] + b[0], -np.inf), np.nextafter(a[1] + b[1], np.inf))
    )


def _interval_scale_nonnegative(a, factor):
    if int(factor) != factor or factor < 0:
        raise ValueError("invalid positive interval scalar")
    if factor == 1:
        return a
    return _cap_checked_interval(
        (np.nextafter(a[0] * factor, -np.inf), np.nextafter(a[1] * factor, np.inf))
    )


def _cap_positive_difference(current, previous):
    low = np.maximum(0, np.nextafter(current[0] - previous[1], -np.inf))
    high = np.minimum(current[1], np.maximum(0, np.nextafter(current[1] - previous[0], np.inf)))
    return _cap_checked_interval((low, high))


def outward_sum(values, mask=None):
    low, high = values if mask is None else (values[0][mask], values[1][mask])
    if not len(low):
        return arb(0)
    u = np.finfo(float).eps / 2
    gamma = len(low) * u / (1 - len(low) * u)
    a = np.nextafter(float(np.sum(low)) - gamma * float(np.sum(np.abs(low))), -np.inf)
    b = np.nextafter(float(np.sum(high)) + gamma * float(np.sum(np.abs(high))), np.inf)
    if not math.isfinite(a) or not math.isfinite(b) or a > b:
        raise ArithmeticError("invalid directed scalar sum")
    return arb(a).union(arb(b))


def _verify_directed_reduction_range(maximum):
    if not 1 <= maximum < 1 << 26:
        raise ValueError("unsupported reduction length")
    u = F(1, 1 << 53)
    gamma = lambda j: j * u / (1 - j * u)
    slack = (1 - u) ** 2 * (1 - gamma(maximum - 1)) * gamma(maximum) - gamma(maximum - 1)
    if slack <= 0:
        raise ArithmeticError("nonpositive directed-reduction slack")
    return slack


def _cap_check_environment():
    info = sys.float_info
    if not __debug__ or (info.radix, info.mant_dig, info.max_exp, info.min_exp, info.rounds) != (
        2,
        53,
        1024,
        -1021,
        1,
    ):
        raise ArithmeticError("assertions and IEEE binary64 round-to-nearest required")
    process = ctypes.CDLL(None)
    process.fegetround.restype = ctypes.c_int
    if process.fegetround() != 0:
        raise ArithmeticError("active rounding mode is not FE_TONEAREST")
    smallest = np.nextafter(np.float64(0), np.float64(1))
    product = np.float64(math.ldexp(1.0, -1022)) * np.float64(math.ldexp(1.0, -52))
    if not 0 < smallest == math.ldexp(1.0, -1074) == product == smallest * np.float64(1):
        raise ArithmeticError("gradual underflow is disabled")


@lru_cache(None)
def _cap_block_partitions(signature):
    if not signature:
        return (((), 1),)
    q, answer = signature[-1], defaultdict(int)
    for blocks, multiplicity in _cap_block_partitions(signature[:-1]):
        answer[tuple(sorted((*blocks, q)))] += multiplicity
        for value, copies in Counter(blocks).items():
            joined = list(blocks)
            joined.remove(value)
            answer[tuple(sorted(joined + [value + q]))] += multiplicity * copies
    return tuple(sorted(answer.items(), key=lambda item: (len(item[0]), item[0])))


def moment_terms(count, signature):
    if count < 0 or any(q < 0 for q in signature):
        raise ValueError("negative moment dimension or exponent")
    return tuple(
        (blocks, m * prod(range(count - len(blocks) + 1, count + 1)))
        for blocks, m in _cap_block_partitions(tuple(sorted(signature)))
        if len(blocks) <= count
    )


@lru_cache(None)
def fiber_splits(signature):
    counts, answer = tuple(sorted(Counter(signature).items())), []
    for chosen in product(*(range(count + 1) for _, count in counts)):
        remaining, exponent, multiplicity = [], 0, 1
        for (power, count), selected in zip(counts, chosen):
            remaining.extend([power] * (count - selected))
            exponent += power * selected
            multiplicity *= comb(count, selected)
        answer.append((tuple(remaining), exponent, multiplicity))
    return tuple(answer)


def cap_leq(left, right):
    return right is None or (left is not None and left <= right)


@dataclass(frozen=True)
class Shell:
    lower: F
    upper: F
    ceiling: F | None


def _cap_aligned_shells(label, step):
    previous, previous_cap, answer = F(0), None, []
    for upper, cap in CAP_SHELL_DATA[label]:
        upper, requested = F(upper), F(cap)
        aligned = (requested // step) * step
        if (
            not previous < upper
            or not 0 < aligned <= requested
            or not cap_leq(aligned, previous_cap)
        ):
            raise ArithmeticError("invalid inward cap shells")
        lower = answer.pop().lower if answer and answer[-1].ceiling == aligned else previous
        answer.append(Shell(lower, upper, aligned))
        previous, previous_cap = upper, aligned
    return tuple(answer)


class CapEngine:
    def __init__(self, intervals=98304, precision=160, fixed_bits=224, arb_threads=1):
        _cap_check_environment()
        if intervals <= 42 or precision < 80 or fixed_bits < 32 or arb_threads < 1:
            raise ValueError("invalid cap engine settings")
        ctx.prec, ctx.threads = precision, arb_threads
        check_flint_signed_fft()
        self.k, self.intervals, self.n = 40, int(intervals), int(intervals) - 40
        self.precision, self.arb_threads, self.fixed_bits = precision, ctx.threads, fixed_bits
        self.S, self.rho_star = F(2742997, 2624989), F(2624989, 10000000)
        self.hq, self.center = self.S / self.intervals, F(9, 10)
        self.h, self.mixture = rational(self.hq), (F(21, 200), F(1, 100), F(907, 5))
        self.descriptors = tuple((s, d) for s in CAP_SIGNATURES for d in range(7))
        self.coefficient_fractions = tuple(F(v, 10**10) for row in CAP_COEFFICIENTS for v in row)
        self.outer_shells = _cap_aligned_shells("outer", self.hq)
        self.geometries = {
            label: dict(
                S=self.S,
                T=F(CAP_SHELL_DATA[label][-1][0]),
                rho=F(262499, 1000000),
                rho_star=self.rho_star,
                outer_shells=self.outer_shells,
                inner_shells=_cap_aligned_shells(label, self.hq),
            )
            for label in LABELS
        }
        finite = sorted(
            {
                s.ceiling
                for g in self.geometries.values()
                for s in g["outer_shells"] + g["inner_shells"]
            }
        )
        self.caps = tuple(finite) + (None,)
        self.outer_masks = tuple(self.shell_mask(s, 40) for s in self.outer_shells)
        self.active_layers = 1 + max(
            self.caps.index(s.ceiling)
            for s, mask in zip(self.outer_shells, self.outer_masks)
            if np.any(mask)
        )
        self.inner_allowed = np.zeros((3, len(self.caps), self.n), dtype=bool)
        for role, label in enumerate(LABELS):
            for shell in self.geometries[label]["inner_shells"]:
                for layer, cap in enumerate(self.caps):
                    if cap_leq(cap, shell.ceiling):
                        self.inner_allowed[role, layer] |= self.shell_mask(shell, 39)
        if np.any(self.inner_allowed[0] & ~self.inner_allowed[1]) or np.any(
            self.inner_allowed[1] & ~self.inner_allowed[2]
        ):
            raise ArithmeticError("cap masks are not nested")
        self.midpoints = tuple(rational((F(j) + F(1, 2)) * self.hq) for j in range(self.n))
        w, slow, fast = map(rational, self.mixture)
        self.root_values = tuple(
            w / (1 + slow * t) + (1 - w) / (1 + fast * t) for t in self.midpoints
        )
        self.Z = sum((g * g for g in self.root_values), arb(0))
        if self.Z.lower() <= 0:
            raise ArithmeticError("nonpositive profile normalization")
        self.coordinate_weights = tuple(g * g / self.Z for g in self.root_values)
        self.normalization, self.physical_scale = arb(40) * self.h / self.Z, (self.h * self.Z) ** 40
        self.full_radial = tuple(
            rational((F(j) + 20) * self.hq - self.center) for j in range(self.n)
        )
        self.directed_reduction_slack = _verify_directed_reduction_range(self.n)
        self._survival, self._midpoint_powers = {}, {}
        (
            self._weighted,
            self._powers,
            self._blocks,
            self._distinct,
            self._moments,
            self._interval_moments,
        ) = {}, {}, {}, {}, {}, {}
        self._affine, self._forms = None, None
        self._square = self._square_polynomials()
        self.square_groups = tuple(
            sorted(self._square.items(), key=lambda item: (sum(item[0]), item[0]))
        )

    def _assert_precision(self):
        if ctx.prec != self.precision or ctx.threads != self.arb_threads:
            raise RuntimeError("global Arb settings changed")

    def shell_mask(self, shell, count):
        indices = np.arange(self.n, dtype=np.int64) + count
        return (indices > int(shell.lower // self.hq)) & (indices <= int(shell.upper // self.hq))

    def aligned_cap(self, cap):
        if cap is None:
            return None
        cap = F(cap)
        aligned = (cap // self.hq) * self.hq
        if not 0 < aligned <= cap:
            raise ValueError("cap contains no positive cell")
        return aligned

    def survival(self, cap):
        self._assert_precision()
        cap = self.aligned_cap(cap)
        if cap not in self._survival:
            if cap is None:
                rows = (arb(1),) * self.n
            else:
                grid = outward_dickman_grid(cap=cap, step=self.hq, count=self.n)
                verify_exact_first_delay(grid)
                rows = [
                    nonnegative((a / self.h).union(b / self.h), arb(1))
                    for a, b in grid.conditional_cell_masses()
                ]
            self._survival[cap] = tuple(nonnegative(v, arb(1)) for v in rows)
        return self._survival[cap]

    def midpoint_power(self, exponent):
        exponent = int(exponent)
        if exponent < 0:
            raise ValueError("negative midpoint power")
        if exponent not in self._midpoint_powers:
            self._midpoint_powers[exponent] = tuple(t**exponent for t in self.midpoints)
        return self._midpoint_powers[exponent]

    def radial_polynomial(self, coefficients):
        coefficients, rows = tuple(map(rational, coefficients)), []
        for x in self.full_radial:
            value = arb(0)
            for coefficient in reversed(coefficients):
                value = value * x + coefficient
            rows.append(value)
        return tuple(rows)

    def weighted(self, cap, exponent):
        self._assert_precision()
        cap, exponent = self.aligned_cap(cap), int(exponent)
        key = cap, exponent
        if key not in self._weighted:
            self._weighted[key] = PositiveFixedPolynomial.from_arb(
                (
                    weight * d * p
                    for weight, d, p in zip(
                        self.coordinate_weights, self.survival(cap), self.midpoint_power(exponent)
                    )
                ),
                self.fixed_bits,
                self.n,
            )
        return self._weighted[key]

    def probability_power(self, cap, count):
        count = int(count)
        if count < 0:
            raise ValueError("negative coordinate count")
        key = self.aligned_cap(cap), count
        if key not in self._powers:
            if count == 0:
                value = PositiveFixedPolynomial.one(self.fixed_bits, self.n)
            elif count == 1:
                value = self.weighted(cap, 0)
            else:
                half = self.probability_power(cap, count // 2)
                value = half.multiply(half)
                if count & 1:
                    value = value.multiply(self.weighted(cap, 0))
            self._powers[key] = value
        return self._powers[key]

    def block_product(self, cap, blocks):
        blocks = tuple(blocks)
        key = self.aligned_cap(cap), blocks
        if key not in self._blocks:
            self._blocks[key] = (
                PositiveFixedPolynomial.one(self.fixed_bits, self.n)
                if not blocks
                else self.weighted(cap, blocks[0])
                if len(blocks) == 1
                else self.block_product(cap, blocks[:-1]).multiply(self.weighted(cap, blocks[-1]))
            )
        return self._blocks[key]

    def moment(self, cap, count, signature):
        self._assert_precision()
        cap, count, signature = (
            self.aligned_cap(cap),
            int(count),
            tuple(sorted(map(int, signature))),
        )
        if count < 0 or min(signature, default=0) < 0:
            raise ValueError("negative moment dimension or exponent")
        key = cap, count, signature
        if key not in self._moments:
            answer = PositiveFixedPolynomial.zero(self.fixed_bits, self.n)
            for blocks, coefficient in moment_terms(count, signature):
                distinct = cap, count, blocks
                if distinct not in self._distinct:
                    self._distinct[distinct] = self.probability_power(
                        cap, count - len(blocks)
                    ).multiply(self.block_product(cap, blocks))
                answer = answer.add(self._distinct[distinct].scale_integer(coefficient))
            self._moments[key] = answer
        return self._moments[key]

    def moment_interval(self, cap, count, signature):
        key = self.aligned_cap(cap), count, tuple(sorted(signature))
        if key not in self._interval_moments:
            self._interval_moments[key] = self.moment(*key).binary64_intervals()
        return self._interval_moments[key]

    def release_moment_caches(self, retain=None):
        for cache in (
            self._weighted,
            self._powers,
            self._blocks,
            self._distinct,
            self._moments,
            self._interval_moments,
        ):
            for key in tuple(cache):
                if retain is None or key[0] != retain:
                    del cache[key]

    def _square_polynomials(self):
        groups = defaultdict(lambda: [F(0)] * 13)
        for i, (si, di) in enumerate(self.descriptors):
            for j in range(i, len(self.descriptors)):
                sj, dj = self.descriptors[j]
                groups[tuple(sorted(si + sj))][di + dj] += (
                    self.coefficient_fractions[i]
                    * self.coefficient_fractions[j]
                    * (1 if i == j else 2)
                )
        return {s: tuple(p) for s, p in groups.items()}

    def frozen_shell_affine(self, progress=False):
        self._assert_precision()
        if self._affine is not None:
            return self._affine
        groups = defaultdict(lambda: [F(0)] * 7)
        for c, (signature, degree) in zip(self.coefficient_fractions, self.descriptors):
            for remaining, exponent, multiplicity in fiber_splits(signature):
                groups[remaining, exponent][degree] += c * multiplicity
        by_remaining = defaultdict(list)
        for (remaining, exponent), polynomial in groups.items():
            if any(polynomial):
                by_remaining[remaining].append((exponent, tuple(polynomial)))
        answer = []
        for shell, mask in zip(self.outer_shells, self.outer_masks):
            rows = {}
            for remaining, terms in by_remaining.items():
                value = (arb(0),) * self.n
                for exponent, polynomial in terms:
                    radial = self.radial_polynomial(polynomial)
                    radial = arb_poly([v if mask[j] else arb(0) for j, v in enumerate(radial)])
                    fiber = [
                        g * d * p
                        for g, d, p in zip(
                            self.root_values,
                            self.survival(shell.ceiling),
                            self.midpoint_power(exponent),
                        )
                    ]
                    correlation = radial * arb_poly(list(reversed(fiber)))
                    value = tuple(a + correlation[self.n - 1 + j] for j, a in enumerate(value))
                rows[remaining] = value
            answer.append(rows)
            if progress:
                print("CAP_AFFINE_SHELL", len(answer), flush=True)
        self._affine = tuple(answer)
        return self._affine

    def denominator(self, progress=False):
        answer = arb(0)
        for i, (shell, mask) in enumerate(zip(self.outer_shells, self.outer_masks)):
            contribution = arb(0)
            for signature, polynomial in self._square.items():
                radial = float_interval(self.radial_polynomial(polynomial))
                moments = self.moment_interval(shell.ceiling, self.k, signature)
                contribution += outward_sum(interval_multiply(radial, moments), mask)
            answer += nonnegative(contribution)
            self.release_moment_caches()
            if progress:
                print("CAP_DENOMINATOR_SHELL", i, contribution, flush=True)
        answer = nonnegative(answer)
        if answer.lower() <= 0:
            raise ArithmeticError("cap denominator is not positive")
        return answer

    def form_scalars(self, progress=False):
        self._assert_precision()
        if self._forms is not None:
            return self._forms
        denominator = self.denominator(progress)
        affine = tuple(
            {s: float_interval(v) for s, v in part.items()}
            for part in self.frozen_shell_affine(progress)
        )
        regions = [arb(0), arb(0), arb(0)]
        for layer, cap in enumerate(self.caps[: self.active_layers]):
            allowed = self.inner_allowed[:, layer]
            masks = allowed[0], allowed[1] & ~allowed[0], allowed[2] & ~allowed[1]
            if not any(np.any(mask) for mask in masks):
                continue
            rows = {}
            for shell, part in zip(self.outer_shells, affine):
                if cap_leq(cap, shell.ceiling):
                    for signature, values in part.items():
                        rows[signature] = (
                            values
                            if signature not in rows
                            else _interval_add(rows[signature], values)
                        )
            if not rows:
                continue
            signatures = tuple(sorted(rows, key=lambda s: (sum(s), s)))
            integrand = np.zeros(self.n), np.zeros(self.n)
            for i, si in enumerate(signatures):
                for j in range(i, len(signatures)):
                    sj = signatures[j]
                    eta = tuple(sorted(si + sj))
                    moments = self.moment_interval(cap, 39, eta)
                    if layer:
                        moments = _cap_positive_difference(
                            moments, self.moment_interval(self.caps[layer - 1], 39, eta)
                        )
                    term = interval_multiply(interval_multiply(rows[si], rows[sj]), moments)
                    integrand = _interval_add(
                        integrand, _interval_scale_nonnegative(term, 1 if i == j else 2)
                    )
            for role, mask in enumerate(masks):
                regions[role] += nonnegative(self.normalization * outward_sum(integrand, mask))
            self.release_moment_caches(retain=cap)
            if progress:
                print("CAP_LAYER", layer, "cap", cap, "regions", regions, flush=True)
        base, plus, tail = map(nonnegative, regions)
        self.release_moment_caches()
        self._forms = dict(
            denominator=denominator,
            J0=base,
            Jplus=plus,
            Jtail=tail,
            J1=base + plus,
            Jfull=base + plus + tail,
            E0=plus + tail,
        )
        return self._forms

    def certify_fixed(
        self,
        mass=F(49999, 50000),
        pair_constant=F(17, 50),
        lambda_value=F(1, 125),
        *,
        progress=False,
    ):
        mass, constant, lam = map(F, (mass, pair_constant, lambda_value))
        if not 0 < lam < mass:
            raise ValueError("invalid hybrid lambda")
        a, b = mass * mass - mass * lam, (1 - mass / lam) * (1 - mass) * constant
        forms = self.form_scalars(progress)
        numerator = forms["J0"] + rational(a + b) * forms["Jplus"] + rational(b) * forms["Jtail"]
        return dict(
            forms=forms,
            numerator=numerator,
            quotient=rational(self.rho_star) * numerator / forms["denominator"],
            hybrid=dict(
                mass=str(mass),
                pair_constant=str(constant),
                **{"lambda": str(lam)},
                a=str(a),
                b=str(b),
            ),
        )


# Complete low, rank-two and third-factorial source contractions.
def positive_enclosure(value):
    value = arb(value)
    if not value.is_finite() or value.upper() < 0:
        raise ArithmeticError("invalid enclosure of a nonnegative quantity")
    return value if value.lower() >= 0 else value.nonnegative_part()


def poly_coefficients(poly, length):
    values = list(poly.coeffs())[:length]
    return values + [arb(0)] * (length - len(values))


def freeze_positive_upper(values):
    result = tuple(value.upper() for value in values)
    if any(not x.is_exact() or not x.is_finite() or x < 0 for x in result):
        raise ArithmeticError("invalid frozen positive coordinate measure")
    return result


@dataclass(frozen=True, slots=True)
class SupportedPositiveFixedPolynomial:
    """Exact leading zeros save work; no small nonzero coefficient is dropped."""

    lower_compact: fmpz_poly
    upper_compact: fmpz_poly
    bits: int
    length: int
    offset: int

    def __post_init__(self):
        if self.bits < 32 or self.length < 1 or not 0 <= self.offset <= self.length:
            raise ValueError("invalid supported fixed-point polynomial")
        remaining = self.length - self.offset
        if max(len(self.lower_compact), len(self.upper_compact)) > remaining:
            raise ValueError("polynomial exceeds its physical truncation")
        if self.offset == self.length:
            if len(self.lower_compact) or len(self.upper_compact):
                raise ArithmeticError("noncanonical zero polynomial")
        elif not len(self.upper_compact) or self.upper_compact[0] <= 0:
            raise ArithmeticError("unproved leading-zero valuation")

    @classmethod
    def zero(cls, bits, length):
        return cls(fmpz_poly([]), fmpz_poly([]), bits, length, length)

    @classmethod
    def one(cls, bits, length):
        return cls(fmpz_poly([1 << bits]), fmpz_poly([1 << bits]), bits, length, 0)

    @classmethod
    def _make(cls, low, high, bits, length, offset):
        low, high = fmpz_poly(low), fmpz_poly(high)
        leading = 0
        while leading < len(high) and not high[leading]:
            if low[leading]:
                raise ArithmeticError("nonzero lower bound below exact support")
            leading += 1
        if leading == len(high):
            if len(low):
                raise ArithmeticError("positive lower bound exceeds zero upper")
            return cls.zero(bits, length)
        offset += leading
        if offset >= length:
            return cls.zero(bits, length)
        remaining = length - offset
        return cls(
            low.right_shift(leading).truncate(remaining),
            high.right_shift(leading).truncate(remaining),
            bits,
            length,
            offset,
        )

    @classmethod
    def from_arb(cls, values, bits, length=None):
        values = tuple(values)
        length = len(values) if length is None else int(length)
        if len(values) > length:
            raise ValueError("input exceeds physical fine-grid length")
        low, high = [], []
        offset = None
        for index, value in enumerate(values):
            value = arb(value)
            if not value.is_finite() or value.upper() < 0:
                raise ArithmeticError("invalid positive coefficient")
            first = max(0, _scaled_dyadic(value.lower(), bits, upper=False))
            last = max(0, _scaled_dyadic(value.upper(), bits, upper=True))
            if first > last:
                raise ArithmeticError("fixed-point endpoints crossed")
            if offset is None:
                if not last:
                    continue
                offset = index
            low.append(first)
            high.append(last)
        return (
            cls.zero(bits, length)
            if offset is None
            else cls(fmpz_poly(low), fmpz_poly(high), bits, length, offset)
        )

    def _compatible(self, other):
        if not isinstance(other, type(self)) or (self.bits, self.length) != (
            other.bits,
            other.length,
        ):
            raise ValueError("incompatible positive fixed-point polynomials")

    def add(self, other):
        self._compatible(other)
        if self.offset == self.length:
            return other
        if other.offset == self.length:
            return self
        offset = min(self.offset, other.offset)
        return type(self)._make(
            self.lower_compact.left_shift(self.offset - offset)
            + other.lower_compact.left_shift(other.offset - offset),
            self.upper_compact.left_shift(self.offset - offset)
            + other.upper_compact.left_shift(other.offset - offset),
            self.bits,
            self.length,
            offset,
        )

    def multiply(self, other):
        self._compatible(other)
        offset = self.offset + other.offset
        if offset >= self.length:
            return type(self).zero(self.bits, self.length)
        remaining, scale = self.length - offset, 1 << self.bits
        low = self.lower_compact.mul_low(other.lower_compact, remaining) // scale
        high = self.upper_compact.mul_low(other.upper_compact, remaining)
        if len(high):
            high = (high + _coefficientwise_ceiling_shift(self.bits, len(high))) // scale
        return type(self)._make(low, high, self.bits, self.length, offset)

    def scale_integer(self, value):
        value = int(value)
        if value < 0:
            raise ValueError("negative positive-polynomial scalar")
        if not value or self.offset == self.length:
            return type(self).zero(self.bits, self.length)
        return type(self)(
            self.lower_compact * value,
            self.upper_compact * value,
            self.bits,
            self.length,
            self.offset,
        )

    def binary64_intervals(self):
        low, high = np.zeros(self.length), np.zeros(self.length)
        for target, values, upper in (
            (low, self.lower_compact.coeffs(), False),
            (high, self.upper_compact.coeffs(), True),
        ):
            target[self.offset : self.offset + len(values)] = np.fromiter(
                (_outward_scaled_positive_integer(v, self.bits, upper=upper) for v in values),
                dtype=np.float64,
                count=len(values),
            )
        np.maximum(low, 0, out=low)
        if not np.isfinite(high).all() or not np.isfinite(low).all() or np.any(low > high):
            raise ArithmeticError("invalid outward binary64 source endpoints")
        return low, high


class _SourceCache(OrderedDict):
    """Eviction recomputes from the identical fixed positive measure."""

    def __init__(self, capacity):
        super().__init__()
        self.capacity = capacity

    def __contains__(self, key):
        found = super().__contains__(key)
        if found:
            self.move_to_end(key)
        return found

    def __getitem__(self, key):
        answer = super().__getitem__(key)
        self.move_to_end(key)
        return answer

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.capacity:
            self.popitem(last=False)


@lru_cache(None)
def _source_coordinate_splits(signature):
    return tuple(sorted(fiber_splits(signature)))


class SourceJets:
    """The ordinary, third-factorial, and two-owner positive moment rings."""

    def __init__(self, raw, locations, bits=192, ring="scalar", marks=None):
        self.ring, self.bits = ring, int(bits)
        self.locations, self.length = tuple(locations), len(locations)
        self.raw = (tuple(raw),) if ring == "scalar" else tuple(map(tuple, raw))
        if ring not in ("scalar", "factorial", "palm"):
            raise ValueError("unknown positive source ring")
        if any(len(row) != self.length for row in self.raw):
            raise ValueError("source law has a different fine-grid length")
        if ring == "palm" and len(self.raw) != 4:
            raise ValueError("two-owner ring needs four channels")
        self.marks = {name: tuple(values) for name, values in (marks or {}).items()}
        if any(len(row) != self.length for row in self.marks.values()):
            raise ValueError("source mark has a different fine-grid length")
        self.base = self._pack(tuple(self._poly(row) for row in self.raw))
        self._weighted = {("ordinary", 0): self.base}
        self._powers = {0: self.one(), 1: self.base}
        scalar = ring == "scalar"
        self._blocks = _SourceCache(48 if scalar else 32)
        self._terms = _SourceCache(48 if scalar else 24)
        self._moments = _SourceCache(32 if scalar else 16)
        self._designated = _SourceCache(24)

    def _poly(self, row):
        return SupportedPositiveFixedPolynomial.from_arb(row, self.bits, self.length)

    def _pack(self, values):
        return values[0] if self.ring == "scalar" else values

    def zero(self):
        return self._pack(
            tuple(SupportedPositiveFixedPolynomial.zero(self.bits, self.length) for _ in self.raw)
        )

    def one(self):
        return self._pack(
            (SupportedPositiveFixedPolynomial.one(self.bits, self.length),)
            + tuple(
                SupportedPositiveFixedPolynomial.zero(self.bits, self.length) for _ in self.raw[1:]
            )
        )

    def add(self, first, second):
        return (
            first.add(second)
            if self.ring == "scalar"
            else tuple(a.add(b) for a, b in zip(first, second))
        )

    def scale(self, values, scalar):
        return (
            values.scale_integer(scalar)
            if self.ring == "scalar"
            else tuple(a.scale_integer(scalar) for a in values)
        )

    def multiply(self, first, second):
        if self.ring == "scalar":
            return first.multiply(second)
        if self.ring == "palm":
            a, x, y, xy = first
            b, u, v, uv = second
            return (
                a.multiply(b),
                x.multiply(b).add(a.multiply(u)),
                y.multiply(b).add(a.multiply(v)),
                xy.multiply(b).add(a.multiply(uv)).add(x.multiply(v)).add(y.multiply(u)),
            )
        result = []
        for degree in range(len(self.raw)):
            value = SupportedPositiveFixedPolynomial.zero(self.bits, self.length)
            for left in range(degree + 1):
                value = value.add(first[left].multiply(second[degree - left]))
            result.append(value)
        return tuple(result)

    def square(self, values):
        if self.ring == "scalar":
            return values.multiply(values)
        if self.ring == "palm":
            a, x, y, xy = values
            return (
                a.multiply(a),
                a.multiply(x).scale_integer(2),
                a.multiply(y).scale_integer(2),
                a.multiply(xy).add(x.multiply(y)).scale_integer(2),
            )
        result = []
        for degree in range(len(values)):
            value = SupportedPositiveFixedPolynomial.zero(self.bits, self.length)
            # Reversed factors give identical fixed endpoints. Add each pair
            # once, then double its exact integer coefficients.
            for left in range(degree // 2 + 1):
                right = degree - left
                term = values[left].multiply(values[right])
                if left != right:
                    term = term.scale_integer(2)
                value = value.add(term)
            result.append(value)
        return tuple(result)

    def weighted(self, exponent, name="ordinary"):
        key = name, int(exponent)
        if exponent < 0 or (name != "ordinary" and name not in self.marks):
            raise ValueError("invalid source power or mark")
        if key not in self._weighted:
            rows = self.raw if name == "ordinary" else (self.marks[name],)
            powers = tuple(location**exponent for location in self.locations)
            self._weighted[key] = self._pack(
                tuple(
                    self._poly([mass * power for mass, power in zip(row, powers)]) for row in rows
                )
            )
        return self._weighted[key]

    def power(self, count):
        if count < 0:
            raise ValueError("negative source coordinate count")
        if count not in self._powers:
            value = self.square(self.power(count // 2))
            if count & 1:
                value = self.multiply(value, self.base)
            self._powers[count] = value
        return self._powers[count]

    def blocks(self, exponents):
        exponents = tuple(sorted(exponents))
        if exponents not in self._blocks:
            self._blocks[exponents] = (
                self.one()
                if not exponents
                else self.weighted(exponents[0])
                if len(exponents) == 1
                else self.multiply(self.blocks(exponents[:-1]), self.weighted(exponents[-1]))
            )
        return self._blocks[exponents]

    def term(self, count, blocks):
        blocks = tuple(sorted(blocks))
        if len(blocks) > count:
            return self.zero()
        key = count, blocks
        if key not in self._terms:
            self._terms[key] = (
                self.power(count)
                if not blocks
                else self.blocks(blocks)
                if len(blocks) == count
                else self.multiply(self.power(count - len(blocks)), self.blocks(blocks))
            )
        return self._terms[key]

    def moment(self, count, signature=()):
        signature = tuple(sorted(signature))
        if count < 0 or min(signature, default=0) < 0:
            raise ValueError("invalid positive angular moment")
        key = count, signature
        if key not in self._moments:
            answer = self.zero()
            for blocks, coefficient in moment_terms(count, signature):
                if coefficient < 0:
                    raise ArithmeticError("negative angular partition coefficient")
                answer = self.add(answer, self.scale(self.term(count, blocks), coefficient))
            self._moments[key] = answer
        return self._moments[key]

    def rows(self, count, signature=(), *, marks=(), channel="both", order=None):
        signature = tuple(sorted(signature))
        if self.ring == "palm":
            return self.moment(count, signature)[
                {"ordinary": 0, "first": 1, "second": 2, "both": 3}[channel]
            ]
        if self.ring == "factorial":
            order = len(self.raw) - 1 if order is None else int(order)
            if not 0 <= order < len(self.raw):
                raise ValueError("factorial order is unavailable")
            return self.moment(count, signature)[order]
        if not marks:
            return self.moment(count, signature)
        if not isinstance(marks, str) or marks not in self.marks:
            raise ValueError("this source ring requires one named witness")
        if count < 1:
            return self.zero()
        key = count, signature, marks
        if key not in self._designated:
            answer = self.zero()
            for rest, exponent, multiplicity in _source_coordinate_splits(signature):
                value = self.moment(count - 1, rest).multiply(self.weighted(exponent, marks))
                answer = answer.add(value.scale_integer(multiplicity))
            self._designated[key] = answer.scale_integer(count)
        return self._designated[key]

    def release_count(self, count):
        for cache in (self._terms, self._moments, self._designated):
            for key in tuple(cache):
                if key[0] == count:
                    del cache[key]


def _low_positive_dickman(size, cutoff, step):
    if cutoff >= 64:
        try:
            return outward_dickman_grid(
                cap=cutoff * step, step=step, count=size - 1, method="trapezoid"
            ).upper
        except ArithmeticError:
            pass  # The positive renewal bound remains valid in the far tail.
    if cutoff < 2:
        raise ValueError("at least two prime cells are required below the cutoff")
    values = [arb(1)] * min(size, cutoff + 1)
    window, floor = arb(cutoff), arb(10) ** (-40)
    for index in range(cutoff + 1, size):
        value = window / index
        if value.lower() <= 0:
            raise ArithmeticError("Dickman renewal recurrence lost positivity")
        if value.upper() <= floor.lower():
            values.extend([floor] * (size - len(values)))
            break
        values.append(value)
        window += value - values[index - cutoff]
    return values


def eulerian_carry_polynomial(variable_count):
    """Exact carry probabilities for uniform fractional cell coordinates."""
    if variable_count < 1:
        raise ValueError("positive Eulerian variable count required")
    values, denominator = [1], 1
    for count in range(2, variable_count + 1):
        previous, values = values, [0] * count
        for index in range(count):
            if index < len(previous):
                values[index] += (index + 1) * previous[index]
            if index:
                values[index] += (count - index) * previous[index - 1]
        denominator *= count
    if sum(values) != denominator:
        raise ArithmeticError("Eulerian carry probabilities do not sum to one")
    return arb_poly([arb(value) / denominator for value in values])


def _source_product(first, second, length):
    return (first * second).truncate(length)


def _low_eulerian_envelopes(high, low, mark, size, minimum, exact_limit=32):
    max_count = (size - 1) // minimum
    exact_count = min(max_count, exact_limit)
    ordinary, designated, term = arb_poly([]), arb_poly([]), arb_poly([1])
    replicated = [arb_poly([]), arb_poly([])]
    for count in range(exact_count + 1):
        ordinary += _source_product(term, eulerian_carry_polynomial(count + 1), size)
        designated += _source_product(term, eulerian_carry_polynomial(count + 2), size)
        if exact_count < max_count:
            for extra in range(2):
                replicated[extra] += _source_product(
                    term, arb_poly([1] * (count + extra + 1)), size
                )
        if count < exact_count:
            term = _source_product(term, high, size) * (arb(1) / (count + 1))
    if exact_count < max_count:
        values = poly_coefficients(high, size)
        unshifted = arb_series(values, size).exp().coeffs()
        shifted = arb_series([arb(0)] + values[:-1], size).exp().coeffs()
        tails = []
        for extra in range(2):
            complete, prefix = [], arb(0)
            for index in range(size):
                if index < len(unshifted):
                    prefix += unshifted[index]
                if index >= extra + 1 and index - extra - 1 < len(shifted):
                    prefix -= shifted[index - extra - 1]
                if prefix.upper() < 0:
                    raise ArithmeticError("positive carry envelope became negative")
                complete.append(prefix.upper())
            tail = [
                (total - initial).upper()
                for total, initial in zip(complete, poly_coefficients(replicated[extra], size))
            ]
            if any(value < 0 for value in tail):
                raise ArithmeticError("positive Eulerian tail lost its upper enclosure")
            tails.append(arb_poly(tail))
        ordinary += tails[0]
        designated += tails[1]
    return (
        _source_product(ordinary, low, size),
        _source_product(_source_product(mark, designated, size), low, size),
    )


def source_low_measures(engine, *, ceiling, order, hard_cap, low, high, slope):
    """The complete Eulerian prefix and a positive all-orders remainder."""
    U, m, z, lo, hi, lam = map(F, (ceiling, order, hard_cap, low, high, slope))
    size, step = engine.intervals, engine.hq
    first, last, cap = int(lo // step), int(-((-hi) // step)), int(-((-z) // step))
    if ctx.prec != engine.precision or m < 2 or not 0 < lo < hi or lam <= 0:
        raise ValueError("invalid low-fragment parameters or precision")
    if first < 2 or not first < last <= cap or last * step > min(z, U / m):
        raise ValueError("low prime bin is under-resolved or admits rank-one failures")
    previous = ctx.cap
    ctx.cap = size
    try:
        dickman = _low_positive_dickman(size, first, step)
        high_cells, marks = [arb(0)] * size, [arb(0)] * size
        for index in range(first, cap):
            high_cells[index] = arb(1) / index
            if index < last:
                marks[index] = high_cells[index]
        decay = -rational(lam)
        low_cells = [
            engine.h * dickman[index] * (decay * rational(index * step)).exp()
            for index in range(size)
        ]
        ordinary, witness = _low_eulerian_envelopes(
            arb_poly(high_cells), arb_poly(low_cells), arb_poly(marks), size, first
        )
    finally:
        ctx.cap = previous
    return SimpleNamespace(
        background=ordinary, designated=witness, high=last * step, low=first * step, cap=cap * step
    )


def _source_low_coordinates(engine, measures, profile_power):
    if profile_power not in (0, 2):
        raise ValueError("source profile power must be zero or two")
    scale = engine.h * (engine.Z if profile_power == 2 else arb(1))
    return tuple(
        freeze_positive_upper(
            mass * root**profile_power / scale
            for mass, root in zip(poly_coefficients(poly, engine.n), engine.root_values)
        )
        for poly in (measures.background, measures.designated)
    )


def source_clipped_group(group, step, witness_upper):
    """Retain every full cell meeting an eligible tier's strict bad event."""
    from copy import deepcopy

    step, upper = F(step), F(witness_upper)
    if group.get("empty"):
        return None
    if step <= 0 or upper <= 0:
        raise ValueError("positive grid step and witness endpoint required")
    eligible = [row for row in group["source_rows"] if F(row["activation"]) < upper]
    if not eligible:
        return None
    order = F(group["order"])
    bound = min(max(F(row["core"]), F(row["ceiling"]) - (order - 1) * upper) for row in eligible)
    cutoff = max(F(group["radial_lower"]), bound)
    dimension = int(group["dimension"])
    pieces = []
    for original in group["aligned_cap_pieces"]:
        first = max(int(original["first_index"]), int(cutoff // step) - dimension + 1)
        if first <= int(original["last_index"]):
            piece = dict(original)
            piece["first_index"] = first
            piece["physical_total_lower"] = str(first * step)
            piece["assigned_radial_lower"] = str(max(F(piece["assigned_radial_lower"]), cutoff))
            pieces.append(piece)
    if not pieces:
        return None
    result = deepcopy(group)
    result.update(source_rows=eligible, aligned_cap_pieces=pieces, radial_lower=str(cutoff))
    return result


class SourceContractions:
    """A single source window, with the global k-coordinate normalization."""

    def __init__(self, engine, group, absolute_weights):
        self.engine, self.group = engine, group
        self.k, self.n = engine.k, engine.n
        self.role, self.dimension = group["role"], int(group["dimension"])
        if self.role not in ("outer", "old_inner", "new_inner"):
            raise ValueError("unknown source role")
        if self.dimension != self.k - (self.role != "outer") or not group["nonlargest"]:
            raise ArithmeticError("source dimension or rank-one exclusion changed")
        cells = set()
        for piece in group["aligned_cap_pieces"]:
            first, last = int(piece["first_index"]), int(piece["last_index"])
            if not 0 <= first <= last < self.n:
                raise ArithmeticError("source cell is outside the physical grid")
            cells.update(range(first, last + 1))
        self.indices = tuple(sorted(cells))
        if not self.indices or group.get("empty"):
            raise ArithmeticError("source cover has inconsistent empty flag")
        zero, one = arb(0), arb(1)
        self.indicator = tuple(one if j in cells else zero for j in range(self.n))
        self.hard_cap = F(group["hard_cap"])
        weights = {
            name: F(absolute_weights[name])
            for name in ("base", "enlarged_minus_base", "outside_enlarged")
        }
        if min(weights.values()) < 0:
            raise ValueError("negative absolute face weight")
        self.maximum_weight = max(weights.values())
        outer_weight = max(weights["enlarged_minus_base"], weights["outside_enlarged"])
        T0, T1 = (F(engine.geometries[name]["T"]) for name in ("base", "enlarged"))
        if T0 > T1:
            raise ArithmeticError("inner radial domains are not nested")
        first_end, second_end = (int(T // engine.hq) - self.k + 1 for T in (T0, T1))
        maximum, outer, outside = map(
            rational, (self.maximum_weight, outer_weight, weights["outside_enlarged"])
        )
        self.face_weights = tuple(
            maximum if j <= first_end else outer if j <= second_end else outside
            for j in range(self.n)
        )

    def affine_prefixes(self, largest_range=None):
        lo, hi = (F(0), self.hard_cap) if largest_range is None else map(F, largest_range)
        hi = min(hi, self.hard_cap)
        if not 0 <= lo <= hi:
            raise ValueError("invalid shared-largest interval")
        shells, parts = self.engine.outer_shells, self.engine.frozen_shell_affine()
        if len(shells) != len(parts):
            raise ArithmeticError("outer affine shell inventory changed")
        answer, running = [], {}
        for length, (shell, part) in enumerate(zip(shells, parts), start=1):
            for signature, values in part.items():
                previous = running.get(signature, (arb(0),) * self.n)
                running[signature] = tuple(a + b for a, b in zip(previous, values))
            current = hi if shell.ceiling is None else min(hi, F(shell.ceiling))
            following = (
                F(0)
                if length == len(shells)
                else F(shells[length].ceiling)
                if shells[length].ceiling is not None
                else hi
            )
            possible = (
                (current > lo if largest_range is not None else current >= 0)
                if length == len(shells)
                else max(lo, following) < current
            )
            if possible:
                answer.append(dict(running))
        return tuple(answer)


def _source_root_square(contractions, moment, radial_multiplier=None):
    engine = contractions.engine
    _verify_directed_reduction_range(engine.n)
    mask = np.zeros(engine.n, dtype=np.bool_)
    mask[np.asarray(contractions.indices, dtype=np.int64)] = True
    multiplier = None
    if radial_multiplier is not None:
        if len(radial_multiplier) != engine.n or any(x.lower() < 0 for x in radial_multiplier):
            raise ValueError("invalid positive full-radius multiplier")
        multiplier = float_interval(radial_multiplier)
        np.maximum(multiplier[0], 0, out=multiplier[0])
    answer = arb(0)
    for signature, polynomial in engine.square_groups:
        radial = float_interval(engine.radial_polynomial(polynomial))
        if multiplier is not None:
            radial = interval_multiply(radial, multiplier)
        answer += outward_sum(
            interval_multiply(radial, moment(signature).binary64_intervals()), mask
        )
    return positive_enclosure(rational(engine.k * contractions.maximum_weight) * answer)


def _source_fiber(contractions, kernel, bits, radial_multiplier=None):
    """W is a function of the full radius BEFORE erasing the extra fiber."""
    window = contractions.indicator
    if len(kernel) != contractions.n:
        raise ValueError("erased fiber has the wrong physical length")
    if radial_multiplier is not None:
        if len(radial_multiplier) != contractions.n:
            raise ValueError("full-radius source multiplier has the wrong length")
        window = tuple(a * b for a, b in zip(window, radial_multiplier))
    indicator = PositiveFixedPolynomial.from_arb(tuple(reversed(window)), bits, contractions.n)
    own = PositiveFixedPolynomial.from_arb(kernel, bits, contractions.n)
    low, high = indicator.multiply(own).binary64_intervals()
    return low[::-1].copy(), high[::-1].copy()


def _source_face_square(contractions, moment_intervals, *, largest_range=None, radial_factor=None):
    engine, outer = contractions.engine, contractions.role == "outer"
    _verify_directed_reduction_range(engine.n)
    weights = float_interval(contractions.face_weights) if outer else None
    mask = None
    if not outer:
        mask = np.zeros(engine.n, dtype=np.bool_)
        mask[np.asarray(contractions.indices, dtype=np.int64)] = True
    if radial_factor is not None:
        if outer or any(len(row) != engine.n for row in radial_factor):
            raise ValueError("invalid inner source radial factor")
        if np.any(radial_factor[0] < 0) or np.any(radial_factor[0] > radial_factor[1]):
            raise ArithmeticError("radial factor is not nonnegative")
    total = arb(0)
    for rows in contractions.affine_prefixes(largest_range):
        affine = {signature: float_interval(values) for signature, values in rows.items()}
        signatures = tuple(sorted(affine, key=lambda sig: (sum(sig), sig)))
        grouped = {}
        for index, left in enumerate(signatures):
            for right in signatures[index:]:
                signature = tuple(sorted(left + right))
                signed = interval_multiply(affine[left], affine[right])
                if left != right:
                    signed = _interval_scale_nonnegative(signed, 2)
                grouped[signature] = (
                    signed
                    if signature not in grouped
                    else _interval_add(grouped[signature], signed)
                )
        subtotal = arb(0)
        for signature, signed in grouped.items():
            if outer:
                signed = interval_multiply(signed, weights)
            elif radial_factor is not None:
                signed = interval_multiply(signed, radial_factor)
            subtotal += outward_sum(interval_multiply(signed, moment_intervals(signature)), mask)
        total += positive_enclosure(subtotal)
    return positive_enclosure(
        arb(engine.k) * engine.h * (engine.h if outer else 1) / engine.Z * total
    )


def source_component_raw(
    engine, original_group, kind, parameters, absolute_weights, fixed_bits=192
):
    """Compute both outer bounds, or the inner bound, without stored endpoints."""
    if fixed_bits < 192 or ctx.prec != engine.precision:
        raise ValueError("invalid source arithmetic precision")
    group = (
        source_clipped_group(original_group, engine.hq, parameters["high"])
        if kind == "low"
        else original_group
    )
    outer = original_group["role"] == "outer"
    zero = {"root_square": arb(0), "outer_face_square": arb(0)} if outer else {"inner_face": arb(0)}
    if group is None or (kind == "high" and F(group["split"]) >= F(group["hard_cap"])):
        return zero
    con = SourceContractions(engine, group, absolute_weights)
    count, multiplier, largest_range = con.dimension, arb(1), None
    radial, inner_radial = None, None
    if kind == "low":
        lo, hi, slope = (F(parameters[name]) for name in ("low", "high", "slope"))
        if not F(group["activation"]) <= lo < hi <= F(group["split"]):
            raise ValueError("low bin is outside the exact source cover")
        measures = source_low_measures(
            engine,
            ceiling=group["ceiling"],
            order=group["order"],
            hard_cap=group["hard_cap"],
            low=lo,
            high=hi,
            slope=slope,
        )
        ordinary, witness = _source_low_coordinates(engine, measures, 2)
        jets = SourceJets(ordinary, engine.midpoints, fixed_bits, marks={"witness": witness})
        offset = (F(group["order"]) - 1) * measures.high + count * engine.hq
        active = set(con.indices)
        rate, ceiling = rational(slope), F(group["ceiling"])
        radial = tuple(
            (rate * rational(j * engine.hq - ceiling + offset)).exp() if j in active else arb(0)
            for j in range(engine.n)
        )
        if outer:
            radial = tuple(value.upper() for value in radial)
        else:
            inner_radial = float_interval(radial)
            np.maximum(inner_radial[0], 0, out=inner_radial[0])
            inactive = np.ones(engine.n, dtype=np.bool_)
            inactive[np.asarray(con.indices, dtype=np.int64)] = False
            inner_radial[0][inactive] = inner_radial[1][inactive] = 0.0
            if not np.isfinite(inner_radial[1]).all() or np.any(inner_radial[0] > inner_radial[1]):
                raise ArithmeticError("inner Chernoff factor escaped binary64")
        selected = lambda d, sig: jets.rows(d, sig, marks="witness")
    elif kind == "rank_two":
        lo, hi = F(parameters["q_low"]), F(parameters["q_high"])
        if not F(group["ceiling"]) / (F(group["order"]) + 1) <= lo < hi <= con.hard_cap:
            raise ValueError("largest-prime bin is outside the source cover")
        source = LargestFragmentBin(lo, hi, F(group["ceiling"]), F(group["order"]))
        fractions = owner_feasible_mass_fractions(engine, source)
        retained = tighten_palm_owner_kernels(
            largest_palm_factory(engine, 2).kernels(source), engine, source, fractions=fractions
        )
        jets = SourceJets(
            (
                retained.ordinary,
                retained.owner,
                retained.nonowner_difference,
                retained.same_owner_difference,
            ),
            engine.midpoints,
            fixed_bits,
            ring="palm",
        )
        selected = lambda d, sig: jets.rows(d, sig, channel="both")
        multiplier = source.mass
        # Only INNER faces condition the retained largest prime on this bin.
        largest_range = None if outer else (lo, hi)
    elif kind == "high":
        retained, _ = factorial_measures(
            engine, minimum=F(group["split"]), hard_cap=con.hard_cap, profile_power=2, degree=3
        )
        jets = SourceJets(retained, engine.midpoints, fixed_bits, ring="factorial")
        selected = lambda d, sig: jets.rows(d, sig, order=3)
    else:
        raise ValueError("unknown source component kind")
    if not outer:
        value = _source_face_square(
            con,
            lambda sig: selected(count, sig).binary64_intervals(),
            largest_range=largest_range,
            radial_factor=inner_radial,
        )
        value = multiplier * value if kind == "rank_two" else 1 * value if kind == "low" else value
        return {"inner_face": positive_enclosure(value)}
    root = _source_root_square(con, lambda sig: selected(count, sig), radial)
    jets.release_count(count)
    if kind == "low":
        erased = _source_low_coordinates(engine, measures, 0)
        plain, witness = (_source_fiber(con, row, fixed_bits, radial) for row in erased)

        @lru_cache(None)
        def mixed(sig):
            return _interval_add(
                interval_multiply(witness, jets.rows(count - 1, sig).binary64_intervals()),
                interval_multiply(
                    plain, jets.rows(count - 1, sig, marks="witness").binary64_intervals()
                ),
            )
    elif kind == "rank_two":
        erased = tighten_palm_owner_kernels(
            largest_palm_factory(engine, 0).kernels(source), engine, source, fractions=fractions
        )
        plain, owner, difference, same = (
            _source_fiber(con, row, fixed_bits)
            for row in (
                erased.ordinary,
                erased.owner,
                erased.nonowner_difference,
                erased.same_owner_difference,
            )
        )

        @lru_cache(None)
        def mixed(sig):
            ordinary, first, second, both = (
                jets.rows(count - 1, sig, channel=channel).binary64_intervals()
                for channel in ("ordinary", "first", "second", "both")
            )
            return _interval_add(
                _interval_add(interval_multiply(same, ordinary), interval_multiply(plain, both)),
                _interval_add(
                    interval_multiply(owner, second), interval_multiply(difference, first)
                ),
            )
    else:
        erased, _ = factorial_measures(
            engine, minimum=F(group["split"]), hard_cap=con.hard_cap, profile_power=0, degree=3
        )
        fibers = tuple(_source_fiber(con, row, fixed_bits) for row in erased)

        @lru_cache(None)
        def mixed(sig):
            answer = None
            for degree in range(4):
                value = interval_multiply(
                    fibers[degree], jets.rows(count - 1, sig, order=3 - degree).binary64_intervals()
                )
                answer = value if answer is None else _interval_add(answer, value)
            return answer

    face = _source_face_square(con, mixed)
    if kind == "rank_two":
        root, face = multiplier * root, multiplier * face
    return {"root_square": positive_enclosure(root), "outer_face_square": positive_enclosure(face)}


def ceil_fraction(value):
    value = F(value)
    return -((-value.numerator) // value.denominator)


def _assert_engine(engine):
    if ctx.prec != engine.precision:
        raise RuntimeError("Arb precision changed during a source calculation")
    if getattr(engine, "test_survival", None) is not None:
        raise ValueError("toy survival laws cannot certify fragment kernels")


@dataclass(frozen=True)
class LargestFragmentBin:
    lower: F
    upper: F
    ceiling: F
    source_order: F

    def __post_init__(self):
        if not 0 < self.lower < self.upper < self.ceiling:
            raise ValueError("invalid largest-prime interval or ceiling")
        if self.source_order < 2:
            raise ValueError("source obstruction order must be at least two")

    @property
    def mass(self):
        return (rational(self.upper) / rational(self.lower)).log().upper()


@dataclass(frozen=True)
class LargestPhysicalKernels:
    ordinary: tuple
    owner: tuple
    nonowner_difference: tuple
    same_owner_difference: tuple


class LargestFragmentPalmFactory:
    """A + u*B_owner + v*(A-B) + u*v*(A_owner-B_owner).

    Here A has cap q and B has cap (U-q)/m.  The coefficient of u*v
    after multiplying coordinates includes same and distinct owners.
    The outside logarithmic q-bin mass is supplied by the caller.
    """

    def __init__(self, *, step, reference, radius, precision_floor=96):
        if step <= 0 or radius <= 0 or len(reference) < 2:
            raise ValueError("positive physical grid and radius required")
        if any(value.lower() < 0 for value in reference):
            raise ValueError("physical-cell reference must be nonnegative")
        self.step, self.radius, self.size = step, radius, len(reference)
        self.reference = tuple(value.upper() for value in reference)
        self.precision_floor = precision_floor
        self._grids = {}

    def _upper_cap_grid(self, cap):
        aligned = ceil_fraction(cap / self.step) * self.step
        key = ("upper", aligned)
        if key not in self._grids:
            self._grids[key] = outward_dickman_grid(cap=aligned, step=self.step, count=self.size)
        return self._grids[key]

    def _lower_cap_grid(self, cap):
        aligned = int(cap / self.step) * self.step
        if aligned <= 0:
            raise ValueError("Dickman lower cap is smaller than one cell")
        key = ("lower", aligned)
        if key not in self._grids:
            self._grids[key] = outward_dickman_grid(cap=aligned, step=self.step, count=self.size)
        return self._grids[key]

    def kernels(self, bin):
        r_min = (bin.ceiling - bin.upper) / bin.source_order
        r_max = (bin.ceiling - bin.lower) / bin.source_order
        same_minimum = bin.lower + r_max
        if min(r_min, r_max) <= 0:
            raise ValueError("nonpositive second-fragment cutoff")
        a = self._upper_cap_grid(bin.upper)
        b_low = self._lower_cap_grid(r_min)
        b_up = self._upper_cap_grid(r_max)
        ordinary, owner, difference, same = [], [], [], []
        q_low = int(bin.lower / self.step)
        q_up = ceil_fraction(bin.upper / self.step)
        for j, weight in enumerate(self.reference):
            # On the full unmarked cell, use A(left,q_upper)-B(right,r_min).
            delta = (a.upper[j] - b_low.lower[min(j + 1, self.size)]).upper()
            if delta < 0:
                delta = arb(0)
            ordinary.append((weight * a.upper[j]).upper())
            difference.append((weight * delta).upper())
            if (j + 1) * self.step <= bin.lower:
                owner.append(arb(0))
                same.append(arb(0))
                continue
            # Residual zero is safe when q crosses this physical cell.
            shift = max(0, j - q_up)
            own_a = a.upper[shift]
            owner.append((weight * b_up.upper[shift]).upper())
            if (j + 1) * self.step <= same_minimum:
                same.append(arb(0))
                continue
            if j * self.step < bin.upper:
                own_b_low = arb(0)
            else:
                shift_low = max(0, j + 1 - q_low)
                own_b_low = b_low.lower[min(shift_low, self.size)]
            own_delta = (own_a - own_b_low).upper()
            if own_delta < 0:
                own_delta = arb(0)
            same.append((weight * own_delta).upper())
        output = LargestPhysicalKernels(
            tuple(ordinary), tuple(owner), tuple(difference), tuple(same)
        )
        if any(value.lower() < 0 for row in output.__dict__.values() for value in row):
            raise ArithmeticError("a positive Palm kernel crossed zero")
        return output


def largest_palm_factory(engine, profile_power):
    _assert_engine(engine)
    if profile_power == 2:
        reference = freeze_positive_upper(g * g / engine.Z for g in engine.root_values)
    elif profile_power == 0:
        reference = (arb(1),) * engine.n
    else:
        raise ValueError("Palm kernels require profile power zero or two")
    return LargestFragmentPalmFactory(
        step=engine.hq, reference=reference, radius=engine.S, precision_floor=engine.precision
    )


_OWNER_MASS_CACHE = OrderedDict()


def owner_feasible_mass_fractions(engine, source):
    """Exact outward fractions of the common logarithmic q-bin mass.

    Ownership needs q<t_right; same-coordinate ownership of the second
    fragment also needs q+(U-q)/m<t_right.  Each ring monomial has exactly
    one owner or same-owner mark, so only those marks receive this factor.
    """
    n, h, precision = int(engine.n), F(engine.hq), int(ctx.prec)
    if n < 1 or h <= 0 or precision < 64:
        raise ValueError("invalid Palm owner grid or precision")
    if hasattr(engine, "precision") and int(engine.precision) != precision:
        raise ArithmeticError("Palm owner fraction uses the wrong precision")
    lo, hi, U, m = map(F, (source.lower, source.upper, source.ceiling, source.source_order))
    if not 0 < lo < hi < U or m <= 1:
        raise ValueError("invalid Palm owner interval or obstruction order")
    key = n, h, lo, hi, U, m, precision
    if key in _OWNER_MASS_CACHE:
        _OWNER_MASS_CACHE.move_to_end(key)
        return _OWNER_MASS_CACHE[key]
    denominator = (rational(hi) / rational(lo)).log()
    if not denominator.is_finite() or denominator.lower() <= 0:
        raise ArithmeticError("Palm logarithmic denominator is not positive")
    zero, one = arb(0), arb(1)

    def upper_fraction(cutoff):
        if cutoff <= lo:
            return zero
        if cutoff >= hi:
            return one
        factor = ((rational(cutoff) / rational(lo)).log() / denominator).upper()
        if not factor.is_finite() or factor < 0:
            raise ArithmeticError("partial Palm logarithmic mass is invalid")
        return min(one, factor)

    owners, same_owners = [], []
    for j in range(n):
        right = (j + 1) * h
        owner_cap = min(hi, right)
        same_cap = min(owner_cap, (m * right - U) / (m - 1))
        owner = upper_fraction(owner_cap)
        same = min(owner, upper_fraction(same_cap))
        if not owner.is_exact() or not same.is_exact():
            raise ArithmeticError("Palm owner fraction was not frozen")
        if not zero <= same <= owner <= one:
            raise ArithmeticError("Palm owner fractions escaped [0,1]")
        owners.append(owner)
        same_owners.append(same)
    result = tuple(owners), tuple(same_owners)
    _OWNER_MASS_CACHE[key] = result
    _OWNER_MASS_CACHE.move_to_end(key)
    while len(_OWNER_MASS_CACHE) > 2:
        _OWNER_MASS_CACHE.popitem(last=False)
    return result


def tighten_palm_owner_kernels(kernels, engine, source, *, fractions=None):
    if fractions is None:
        fractions = owner_feasible_mass_fractions(engine, source)
    n = int(engine.n)
    if (
        not isinstance(fractions, tuple)
        or len(fractions) != 2
        or any(len(row) != n for row in fractions)
    ):
        raise ValueError("Palm owner-fraction arrays have the wrong length")
    zero, one = arb(0), arb(1)
    for owner, same in zip(*fractions):
        if (
            not isinstance(owner, arb)
            or not isinstance(same, arb)
            or not owner.is_exact()
            or not same.is_exact()
            or not owner.is_finite()
            or not same.is_finite()
            or not zero <= same <= owner <= one
        ):
            raise ValueError("invalid supplied Palm owner fractions")
    ordinary, difference = kernels.ordinary, kernels.nonowner_difference
    if any(
        len(row) != n
        for row in (ordinary, kernels.owner, difference, kernels.same_owner_difference)
    ):
        raise ValueError("Palm kernel rows have incompatible lengths")
    owner = freeze_positive_upper(
        value * factor for value, factor in zip(kernels.owner, fractions[0])
    )
    same = freeze_positive_upper(
        value * factor for value, factor in zip(kernels.same_owner_difference, fractions[1])
    )
    return type(kernels)(ordinary, owner, difference, same)


def eulerian_factorial_cell_jets(
    *,
    high: arb_poly,
    low: arb_poly,
    radial_limit: int,
    minimum_high_index: int,
    factorial_degree: int = 3,
    freeze_upper_endpoints: bool = True,
) -> list[list[arb]]:
    """Positive physical-cell upper measures for factorial orders 0..d.

    Freeze every row to exact upper endpoints before contracting a signed
    angular square. Without freezing, the returned intervals are not fixed
    measures for signed algebra.
    """
    if radial_limit < 1 or minimum_high_index < 1 or factorial_degree < 0:
        raise ValueError("positive grid and nonnegative factorial degree required")

    def multiply(first: arb_poly, second: arb_poly) -> arb_poly:
        return _source_product(first, second, radial_limit)

    max_count = (radial_limit - 1) // minimum_high_index
    active_degree = min(factorial_degree, max_count)
    backgrounds = [arb_poly([]) for _ in range(active_degree + 1)]
    carries = [eulerian_carry_polynomial(count + 1) for count in range(max_count + 1)]
    designated = []
    ordinary = arb_poly([1])
    # Share H^r/r! between rows, keeping each row's original addition order.
    for extra in range(max_count + 1):
        if extra <= active_degree:
            designated.append(ordinary)
        for marked in range(min(active_degree, max_count - extra) + 1):
            backgrounds[marked] += multiply(ordinary, carries[extra + marked])
        if extra < max_count:
            ordinary = multiply(ordinary, high) * (arb(1) / (extra + 1))

    answer = []
    for marked in range(factorial_degree + 1):
        if marked > active_degree:
            coefficients = [arb(0)] * radial_limit
        else:
            total = multiply(multiply(designated[marked], backgrounds[marked]), low)
            coefficients = poly_coefficients(total, radial_limit)
            if freeze_upper_endpoints:
                coefficients = [value.upper() for value in coefficients]
            if any(value.lower() < 0 for value in coefficients):
                raise ArithmeticError("a positive factorial Eulerian cell crossed zero")
        answer.append(coefficients)
    return answer


def factorial_measures(engine, *, minimum, hard_cap, profile_power, degree=3):
    """Positive normalized E[binom(N_high,r)] coordinate laws, r<=degree.

    Floor the low cutoff and ceil the high cap. High prime cell j has
    dominating integrated mass 1/j; the low-total cell has mass h*rho_upper.
    Divide its physical output by h*Z (profile 2), or h (erased profile 0).
    """
    _assert_engine(engine)
    p0, z = map(F, (minimum, hard_cap))
    low_index = int(p0 // engine.hq)
    cap_index = ceil_fraction(z / engine.hq)
    if not 2 <= low_index < cap_index <= engine.intervals or degree < 0:
        raise ValueError("invalid factorial source cutoff")
    if profile_power not in (0, 2):
        raise ValueError("unsupported factorial profile power")
    high = [arb(0)] * engine.intervals
    for j in range(low_index, cap_index):
        high[j] = arb(1) / j
    smooth = outward_dickman_grid(cap=low_index * engine.hq, step=engine.hq, count=engine.intervals)
    low = [engine.h * smooth.upper[j] for j in range(engine.intervals)]
    physical = eulerian_factorial_cell_jets(
        high=arb_poly(high),
        low=arb_poly(low),
        radial_limit=engine.intervals,
        minimum_high_index=low_index,
        factorial_degree=degree,
        freeze_upper_endpoints=True,
    )
    scale = engine.h * (engine.Z if profile_power == 2 else arb(1))
    profile = [root**profile_power for root in engine.root_values]
    normalized = tuple(
        freeze_positive_upper(row[j] * profile[j] / scale for j in range(engine.n))
        for row in physical
    )
    return normalized, dict(
        kind="positive_eulerian_factorial",
        degree=degree,
        requested_minimum=str(p0),
        outward_minimum=str(low_index * engine.hq),
        requested_cap=str(z),
        outward_cap=str(cap_index * engine.hq),
        profile_power=profile_power,
    )


# Fresh evaluation, exact outward rounding and final inequality.
def _driver_ceil(value):
    return math.ceil(F(value))


def _driver_endpoint(value):
    if not value.is_finite() or not value.is_exact():
        raise ArithmeticError("a reported endpoint is not a finite exact dyadic")
    mantissa, exponent = map(int, value.man_exp())
    return F(mantissa * (1 << exponent)) if exponent >= 0 else F(mantissa, 1 << -exponent)


def _driver_interval(value, positive=False):
    if not value.is_finite():
        raise ArithmeticError("nonfinite numerical result")
    lower, upper = _driver_endpoint(value.lower()), _driver_endpoint(value.upper())
    if lower > upper or (positive and lower < 0):
        raise ArithmeticError("invalid outward numerical endpoints")
    return dict(lower=str(lower), upper=str(upper))


def _driver_emit(event, **fields):
    print(json.dumps(dict(event=event, **fields), sort_keys=True, allow_nan=False), flush=True)


def _driver_key(task):
    return task["group"], task["kind"], int(task["index"])


def _driver_inventory():
    keys = [_driver_key(task) for task in TASKS]
    if len(TASKS) != 97 or len(set(keys)) != 97 or len(GROUP_BY_ID) != 6:
        raise ArithmeticError("incomplete or duplicate source task inventory")
    if set(GROUP_BY_ID) != {g["id"] for g in SOURCE_GROUPS}:
        raise ArithmeticError("source group inventories disagree")
    if sum(2 if GROUP_BY_ID[t["group"]]["role"] == "outer" else 1 for t in TASKS) != 149:
        raise ArithmeticError("the source schedule does not contain 149 raw forms")
    for group in SOURCE_GROUPS:
        for kind, left, right in (
            ("low", group["activation"], group["split"]),
            ("rank_two", group["rank_two_guard"]["largest_lower"], group["hard_cap"]),
        ):
            rows = sorted(
                (t for t in TASKS if t["group"] == group["id"] and t["kind"] == kind),
                key=lambda t: t["index"],
            )
            lower, upper = ("low", "high") if kind == "low" else ("q_low", "q_high")
            end = F(left)
            for index, task in enumerate(rows):
                p = task["parameters"]
                if task["index"] != index or F(p[lower]) != end or not end < F(p[upper]):
                    raise ArithmeticError("source prime intervals do not tile their required range")
                end = F(p[upper])
                if kind == "low" and F(p["slope"]) <= 0:
                    raise ArithmeticError("nonpositive Chernoff slope")
            if end != F(right):
                raise ArithmeticError("source prime cover ends before its required endpoint")
        high = [t for t in TASKS if t["group"] == group["id"] and t["kind"] == "high"]
        if len(high) != 1 or high[0]["index"] != 0 or high[0]["parameters"]:
            raise ArithmeticError("missing or duplicated third-factorial source component")
    return {_driver_key(task): task for task in TASKS}


def _driver_engine(source=False):
    bits = POLICY["source_fractional_bits" if source else "cap_fractional_bits"]
    threads = POLICY["production_source_threads" if source else "production_cap_threads"]
    engine = CapEngine(
        intervals=POLICY["intervals"],
        precision=POLICY["arb_precision_bits"],
        fixed_bits=bits,
        arb_threads=threads,
    )
    if (
        engine.k != 40
        or engine.n != POLICY["convolution_length"]
        or engine.hq != F(DERIVED_INPUTS["layout"]["grid_step"])
        or engine.rho_star != F(2624989, 10**7)
        or engine.fixed_bits != bits
        or engine.arb_threads != threads
    ):
        raise ArithmeticError("cap/source engine does not use the exact common trial and grid")
    return engine


def compute_fresh_cap():
    """Enclose the cap forms and round their endpoints outwards to decimal units."""
    engine = _driver_engine()
    hybrid = DERIVED_INPUTS["hybrid"]
    result = engine.certify_fixed(
        F(hybrid["mass"]), F(hybrid["pair_constant"]), F(hybrid["lambda"]), progress=True
    )
    record = dict(
        normalized_forms={
            key: _driver_interval(value, True) for key, value in result["forms"].items()
        },
        hybrid_numerator=_driver_interval(result["numerator"]),
        hybrid_quotient=_driver_interval(result["quotient"]),
        normalization_factor=_driver_interval(engine.physical_scale, True),
    )
    scale = POLICY["cap_decimal_scale"]
    interval = record["normalized_forms"]["denominator"]
    lo, hi = F(interval["lower"]), F(interval["upper"])
    numerator = F(record["hybrid_numerator"]["lower"])
    units = dict(
        I_lower=(lo * scale) // 1,
        I_upper=_driver_ceil(hi * scale),
        J_lower=(numerator * scale) // 1,
    )
    if units["I_lower"] <= 0 or units["I_lower"] > units["I_upper"]:
        raise ArithmeticError("the freshly rounded cap denominator is not positive")
    record["rounded_units"] = units
    return record


_DRIVER_SOURCE_ENGINE = None


def compute_source_task(task):
    """One engine per spawned process; every task recomputes its source law."""
    global _DRIVER_SOURCE_ENGINE
    started = monotonic()
    print("SOURCE_START", *_driver_key(task), file=sys.stderr, flush=True)
    # Lazy construction propagates initialization failures as task errors;
    # a Pool initializer that raises could otherwise be restarted repeatedly.
    if _DRIVER_SOURCE_ENGINE is None:
        _DRIVER_SOURCE_ENGINE = _driver_engine(source=True)
    group = GROUP_BY_ID[task["group"]]
    raw = source_component_raw(
        _DRIVER_SOURCE_ENGINE,
        group,
        task["kind"],
        task["parameters"],
        DERIVED_INPUTS["hybrid"]["outer_absolute_weights"],
        POLICY["source_fractional_bits"],
    )
    expected = {"root_square", "outer_face_square"} if group["role"] == "outer" else {"inner_face"}
    if set(raw) != expected:
        raise ArithmeticError("source component returned the wrong raw forms")
    record = dict(
        task=task,
        raw_forms={key: _driver_interval(value, True) for key, value in raw.items()},
        elapsed_seconds=monotonic() - started,
    )
    del raw
    gc.collect()  # Retain affine/profile rows, but release this task's positive jets.
    return record


def round_source_component(record, denominator):
    """All rounding uses the fresh cap floor, never a published denominator."""
    task, denominator = record["task"], F(denominator)
    if denominator <= 0:
        raise ArithmeticError("positive fresh source normalization required")
    raw_scale, component_scale = (
        POLICY[name] for name in ("raw_relative_decimal_scale", "component_relative_decimal_scale")
    )
    relative = {}
    for key, bounds in record["raw_forms"].items():
        lower, upper = F(bounds["lower"]), F(bounds["upper"])
        if not 0 <= lower <= upper:
            raise ArithmeticError("invalid positive source raw endpoint")
        relative[key] = _driver_ceil(upper * raw_scale / denominator)
    role = GROUP_BY_ID[task["group"]]["role"]
    if role == "outer":
        c = F(task["young_q"], POLICY["young_denominator"])
        if (
            c <= 0
            or c != F(task["young"])
            or task["young_denominator"] != POLICY["young_denominator"]
        ):
            raise ArithmeticError("invalid fixed Young parameter")
        cost = (
            c * F(relative["root_square"], raw_scale)
            + F(relative["outer_face_square"], raw_scale) / c
        )
    else:
        coefficient = F(DERIVED_INPUTS["hybrid"]["d0" if role == "old_inner" else "one_minus_b"])
        if coefficient <= 0 or coefficient != F(task["restoration_coefficient"]):
            raise ArithmeticError("incorrect inner restoration coefficient")
        cost = coefficient * F(relative["inner_face"], raw_scale)
    units = _driver_ceil(cost * component_scale)
    return dict(
        record,
        raw_relative_units=relative,
        component_relative_units=units,
        normalized_loss_upper=str(denominator * F(units, component_scale)),
    )


def assemble_fresh_certificate(cap, rows, workers, seconds):
    """Verify full source coverage and evaluate the exact final margin."""
    expected = _driver_inventory()
    by_key = {}
    for row in rows:
        key = _driver_key(row["task"])
        if key not in expected or row["task"] != expected[key] or key in by_key:
            raise ArithmeticError("a source result is missing, duplicated, or from another task")
        by_key[key] = row
    if set(by_key) != set(expected) or sum(len(row["raw_forms"]) for row in rows) != 149:
        raise ArithmeticError("not all 97 source components and 149 raw forms were evaluated")
    rows = [by_key[_driver_key(task)] for task in TASKS]
    scale = POLICY["component_relative_decimal_scale"]
    denominator = F(cap["rounded_units"]["I_lower"], POLICY["cap_decimal_scale"])
    totals = []
    for group in SOURCE_GROUPS:
        selected = [row for row in rows if row["task"]["group"] == group["id"]]
        totals.append(
            dict(
                group=group["id"],
                counts=dict(Counter(r["task"]["kind"] for r in selected)),
                component_relative_units=sum(r["component_relative_units"] for r in selected),
            )
        )
    total = sum(group["component_relative_units"] for group in totals)
    loss = denominator * F(total, scale)
    upper = F(cap["rounded_units"]["I_upper"], POLICY["cap_decimal_scale"])
    numerator = F(cap["rounded_units"]["J_lower"], POLICY["cap_decimal_scale"])
    rho = F(2624989, 10**7)
    margin = rho * (numerator - loss) / upper - 1
    passed = margin > F(1, 50000)
    return dict(
        status="PASS_FRESH_NUMERICAL_CERTIFICATE" if passed else "FAIL_NUMERICAL_MARGIN",
        settings=dict(POLICY, workers=workers, multiprocessing_start_method="spawn"),
        normalization="physical form / (h*sum(g_j^2))^40",
        cap=cap,
        source_normalization_denominator=str(denominator),
        components=rows,
        group_totals=totals,
        component_count=len(rows),
        raw_form_count=149,
        source_total_relative_units=total,
        normalized_source_loss_upper=str(loss),
        final_quotient_lower=str(1 + margin),
        final_margin_lower=str(margin),
        required_margin="1/50000",
        passed=passed,
        elapsed_seconds=seconds,
    )


def main():
    parser = argparse.ArgumentParser(description="Recompute the full 186 numerical certificate.")
    parser.add_argument(
        "--workers",
        type=int,
        choices=(1, 4),
        default=1,
        help="number of source worker processes (default: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("prime_gap_186_fresh.json"),
        help="new JSON receipt; an existing file is never overwritten",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to replace an existing receipt")
    _driver_inventory()
    started = monotonic()
    _driver_emit("CAP_START")
    cap = compute_fresh_cap()
    gc.collect()  # The serial cap engine is gone before any source process starts.
    denominator = F(cap["rounded_units"]["I_lower"], POLICY["cap_decimal_scale"])
    _driver_emit("CAP_COMPLETE", cap=cap, source_normalization_denominator=str(denominator))
    rows = []
    # Each process imports this same script normally and starts with fresh FLINT
    # settings. Pool termination also stops remaining work if any task fails.
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for record in pool.imap_unordered(compute_source_task, TASKS, chunksize=1):
            row = round_source_component(record, denominator)
            rows.append(row)
            _driver_emit("SOURCE_COMPLETE", completed=len(rows), required=len(TASKS), component=row)
    receipt = assemble_fresh_certificate(cap, rows, args.workers, monotonic() - started)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(receipt, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    _driver_emit(
        "FINAL",
        status=receipt["status"],
        group_totals=receipt["group_totals"],
        final_margin_lower=receipt["final_margin_lower"],
    )
    if not receipt["passed"]:
        raise ArithmeticError("the freshly computed margin does not exceed 1/50000")


if __name__ == "__main__":
    main()
