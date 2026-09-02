# Prime Gaps at Most 186

This repository contains a Lean 4 formalization of a prime-gap bound and a
Python numerical certificate. The Lean results remain **conditional on three explicit input axioms**;
the cited mathematical estimates and numerical computations have not been
turned into Lean proofs of those inputs.

## The result

For the sequence of primes $p_n$, the target bound is

```math
\liminf_{n\to\infty}(p_{n+1}-p_n)\le 186.
```

The development derives $\mathrm{DHL}[40,2]$ from the inputs below: every
admissible set of forty integer shifts has infinitely many translates containing
at least two primes. Admissibility means omitting a residue class modulo every
prime. Applying this to the included tuple of diameter 186 gives the gap bound.

The main declarations in [PrimeGaps186.lean](PrimeGaps186.lean), in namespace
`PrimeGap186`, are:

| Declaration | Result |
| --- | --- |
| `dhl_40_2` | $\mathrm{DHL}[40,2]$ for every admissible integer tuple. |
| `infinite_two_prime_translates_admissibleTuple` | Infinitely many two-prime translates of the explicit tuple. |
| `primeGapLiminf_le_186` | The consecutive-prime gap bound. |

## Assumed Deligne-type estimates

For a prime $p$, write $e_p(x)=\exp(2\pi i\widetilde{x}/p)$, where
$\widetilde{x}$ is any integer representative of $x\in\mathbb{F}_p$. Define

```math
\mathrm{Kl}_3(c;p)
=\frac1p\sum_{\substack{x_1,x_2,x_3\in\mathbb{F}_p\\x_1x_2x_3=c}}
e_p(x_1+x_2+x_3),
```

```math
K_2(c;p)=\sum_{u\in\mathbb{F}_p^\times}e_p(u+c/u).
```

The axiom `PrimeGap186.kloosterman3_bound` assumes the following bound for
**every prime** $p$ and all $c\in\mathbb{F}_p^\times$:

```math
\left|\mathrm{Kl}_3(c;p)\right|\le 3.
```

This follows from Deligne's theorem as stated in Nicholas M. Katz,
[*Gauss Sums, Kloosterman Sums, and Monodromy Groups*, Annals of Mathematics
Studies 116, Princeton University Press (1988), Theorem 4.1.1(1)–(2), p. 49](https://web.math.princeton.edu/~nmk/Katz-GKM.pdf#page=29).
With $n=3$, trivial multiplicative characters, and $b_1=b_2=b_3=1$, rank three
and weight two give the raw bound $3p$; our normalization divides by $p$.

The axiom `PrimeGap186.kloosterman2_correlation_bound` assumes the following
bound for **every prime** $p$ and all $A,B\in\mathbb{F}_p^\times$:

```math
\left|\sum_{t\in\mathbb{F}_p\setminus\{0,-1\}}
K_2(A/t;p)\,K_2(B/(t+1);p)\right|\le 8p\sqrt p.
```

This is Étienne Fouvry, Emmanuel Kowalski, and Philippe Michel,
[*The Friedlander–Iwaniec character sum*, 14 June 2013, Proposition 2, p. 1](https://people.math.ethz.ch/~kowalski/friedlander-iwaniec-sum.pdf#page=1).
Their normalized $\mathrm{Kl}_2(c)$ equals $K_2(c;p)/\sqrt p$ after
inverting the summation variable, so their $8\sqrt p$ bound becomes
$8p\sqrt p$ here. No condition $A\ne B$ is imposed; the two poles are excluded
even when $A=B$.

These estimates are established in the cited literature, but remain unproved
inputs in this Lean development.

## Numerical input and certificate

`PrimeGap186.physical_integral_bounds` assumes 104 outer and 45 inner
physical-integral upper bounds, plus three cap bounds.

The [Python certificate](prime_gap_186_certificate.py) recomputes the trial
from scratch. The tested environment used Python 3.12.13, NumPy 2.2.6,
python-flint 0.9.0, and a custom FLINT 3.6.0 build with corrected signed
polynomial convolution (not bundled).

```sh
python3 -B prime_gap_186_certificate.py --workers 4 --output prime_gap_186_fresh.json
```

Use a new output path. Keep `PYTHONOPTIMIZE` unset and do not use `-O` or `-OO`.
Mandatory floating-point and signed-convolution checks must pass. A successful
run produces a receipt with `passed: true`; it does not discharge any Lean
axiom.

## Building and verification

The project pins Lean 4.34.0-rc2 and its Mathlib dependencies. With
[elan](https://github.com/leanprover/elan) installed, run:

```sh
lake exe cache get
lake build PrimeGaps186
```

The registered Lean build passed without errors or warnings. Comparator matched
all three results to `Challenge.lean`, and Nanoda and Lean’s kernel accepted their
proofs in a local Colima Linux VM. The [configuration](comparator/main.json) permits
the three documented project axioms plus `propext`, `Quot.sound`, and
`Classical.choice` (six total); this verifies conditional proofs, not the inputs
themselves. The numerical certificate is unchanged from its earlier passing run.

[Challenge.lean](Challenge.lean) specifies the statements and input assumptions,
with three intentional theorem placeholders. See the
[Comparator instructions](comparator/README.md) and
[formalization metadata](formalization.yaml) for the checking setup and status.

Project contributions use [Apache 2.0](LICENSE); existing third-party notices
remain applicable.
