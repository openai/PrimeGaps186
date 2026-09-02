import Mathlib

/-!
# Conditional prime-gap comparison statements

This reference was written after the solution; it is not an independent pre-solution
challenge. Its three theorem `sorry` bodies are intentional specification placeholders.
The proof module must never import `Challenge`.

The three project axioms below are unproved inputs. Their types and the fixed bodies of every
supporting definition are copied from `PrimeGaps186.lean`; there are no definition holes.
Comparator checks these inputs against the solution as well as the three target statements.
A successful run verifies the solution conditionally on these inputs, not the inputs themselves.
See `comparator/main.json` for the exact permitted axiom names.
-/

section

open scoped BigOperators ENNReal NNReal Topology BoundedContinuousFunction
open scoped MeasureTheory.BoundedContinuousFunction Pointwise

open MeasureTheory AddChar Filter Metric

namespace PrimeGap186

open Classical in
/--
The rank-three Kloosterman sum over `ZMod p`, normalized by `1 / p`. It sums the standard
additive character of `u + v + w` over triples with product `c`.
-/
noncomputable def normalizedKloosterman3 (p : ℕ) [Fact p.Prime]
    (c : ZMod p) : ℂ :=
  (p : ℂ)⁻¹ *
    ∑ u : ZMod p, ∑ v : ZMod p, ∑ w : ZMod p,
      if u * v * w = c then ZMod.stdAddChar (u + v + w) else 0

/--
The unnormalized classical Kloosterman sum `∑ u ≠ 0, ψ(u + c / u)` over the prime field, indexed
by its units.
-/
noncomputable def unnormalizedKloosterman2 (p : ℕ) [Fact p.Prime]
    (c : ZMod p) : ℂ :=
  ∑ u : (ZMod p)ˣ,
    ZMod.stdAddChar ((u : ZMod p) + c / (u : ZMod p))

end PrimeGap186

axiom PrimeGap186.kloosterman3_bound :
  ∀ (p : ℕ) [Fact p.Prime] (c : ZMod p),
    c ≠ 0 → ‖PrimeGap186.normalizedKloosterman3 p c‖ ≤ (3 : ℝ)

axiom PrimeGap186.kloosterman2_correlation_bound :
  ∀ (p : ℕ) [Fact p.Prime] (A B : ZMod p),
    A ≠ 0 → B ≠ 0 →
      ‖∑ t : ZMod p, if t ≠ 0 ∧ t ≠ -1 then
        PrimeGap186.unnormalizedKloosterman2 p (A / t) *
          PrimeGap186.unnormalizedKloosterman2 p (B / (t + 1)) else 0‖ ≤
        8 * (p : ℝ) * Real.sqrt (p : ℝ)

namespace PrimeGap186

/--
The finite atomic measure assigning weight `max (x i) 0` to each sample `x i`. Repeated sample
locations contribute additively, and negative samples have zero weight.
-/
noncomputable def weightedEmpirical (n : ℕ) (x : Fin n → ℝ) : FiniteMeasure ℝ :=
  ∑ i : Fin n,
    let atom : FiniteMeasure ℝ := ⟨Measure.dirac (x i), inferInstance⟩
    (x i).toNNReal • atom

/--
The pushforward obtained by drawing a Poisson count of mean `μ.mass`, drawing locations from
`μ.normalize`, and forming their weighted empirical measure. The atoms are weighted by their
nonnegative locations, not by unit mass.
-/
noncomputable def finitePoissonLaw (μ : FiniteMeasure ℝ) : Measure (FiniteMeasure ℝ) :=
  Measure.map
    (fun p : ℕ × (ℕ → ℝ) =>
      weightedEmpirical p.1 (fun i : Fin p.1 => p.2 i.val))
    ((ProbabilityTheory.poissonMeasure μ.mass).prod
      (Measure.infinitePi (fun _ : ℕ => (μ.normalize : Measure ℝ))))

/--
The finite measure with density `1 / u` on `(0, ζ] ∩ (2^k, 2^(k+1)]`. The dyadic lower bound
keeps the density integrable on each band.
-/
noncomputable def cappedDyadicIntensity (ζ : ℝ) (k : ℤ) : FiniteMeasure ℝ :=
  ⟨((volume.restrict (Set.Ioc (0 : ℝ) ζ)).withDensity
      (fun u : ℝ => ENNReal.ofReal (1 / u))).restrict
      (Set.Ioc ((2 : ℝ) ^ k) ((2 : ℝ) ^ (k + 1))), by
    rw [restrict_withDensity measurableSet_Ioc]
    apply isFiniteMeasure_withDensity
    apply ne_of_lt
    calc
      (∫⁻ u, ENNReal.ofReal (1 / u)
          ∂((volume.restrict (Set.Ioc (0 : ℝ) ζ)).restrict
            (Set.Ioc ((2 : ℝ) ^ k) ((2 : ℝ) ^ (k + 1)))))
        ≤ ∫⁻ _u, ENNReal.ofReal (1 / (2 : ℝ) ^ k)
            ∂((volume.restrict (Set.Ioc (0 : ℝ) ζ)).restrict
              (Set.Ioc ((2 : ℝ) ^ k) ((2 : ℝ) ^ (k + 1)))) := by
          apply lintegral_mono_ae
          filter_upwards [ae_restrict_mem measurableSet_Ioc] with u hu
          exact ENNReal.ofReal_le_ofReal
            (one_div_le_one_div_of_le (zpow_pos (by norm_num) k) hu.1.le)
      _ < ∞ := by
        rw [lintegral_const]
        exact ENNReal.mul_lt_top ENNReal.ofReal_lt_top (measure_lt_top _ _)⟩

/--
Sum a doubly infinite family of fragment measures when the resulting measure has finite mass;
return the zero measure otherwise. The fallback makes this a total function into finite
measures.
-/
noncomputable def finiteFragments (ω : ℤ → FiniteMeasure ℝ) : FiniteMeasure ℝ := by
  classical
  exact if h : IsFiniteMeasure (Measure.sum (fun k : ℤ => (ω k : Measure ℝ))) then
      ⟨Measure.sum (fun k : ℤ => (ω k : Measure ℝ)), h⟩
    else 0

/-- For positive `ζ`, the probability law of the finite weighted fragment measure.
Finite total mass does not mean finitely many fragments. The paper's measure `ν_ζ`
is `exp(γ) * ζ` times this law. -/
noncomputable def fragmentLaw (ζ : ℝ) : Measure (FiniteMeasure ℝ) :=
  Measure.map finiteFragments
    (Measure.infinitePi (fun k : ℤ => finitePoissonLaw (cappedDyadicIntensity ζ k)))

/--
An outer-table row `(scale, rootBound, faceBound, budget)`. The scale is encoded in units of
`10⁻⁶`, the two component bounds in units of `10⁻¹⁸`, and the upward-rounded combined budget in
units of `10⁻¹²`; the component bounds apply after division by the reference value `23685317816
/ 10^24`.
-/
abbrev OuterBoundRow := ℕ × ℕ × ℕ × ℕ

/--
An inner-table row `(massBound, budget)`. The component bound, after division by the reference
value `23685317816 / 10^24`, is encoded in units of `10⁻¹⁸`; the upward-rounded weighted budget
is encoded in units of `10⁻¹²`.
-/
abbrev InnerBoundRow := ℕ × ℕ

/--
The 17 exact integer rows for the order-two outer-component bounds and their rounded budgets.
-/
def outerOrderTwoBounds : List OuterBoundRow :=
  [(961904, 11, 10, 1),
   (502424, 2285, 577, 1),
   (483341, 11432060, 2670744, 12),
   (547373, 3056104728, 915663654, 3346),
   (563915, 37877639997, 12045112668, 42720),
   (583181, 300901046806, 102336788484, 350961),
   (604629, 2682803914309, 980771899210, 3244207),
   (620671, 3338737765461, 1286194297547, 4144522),
   (629321, 7260461043003, 2875471614189, 9138326),
   (635211, 1211995036896, 489032601185, 1539747),
   (616326, 8286469691008, 3147682021553, 10214338),
   (593862, 4616001082128, 1627937050440, 5482540),
   (573977, 2353287968619, 775291485464, 2701470),
   (553178, 1146587714775, 350863740368, 1268537),
   (531463, 529511465762, 149562603056, 562833),
   (508862, 229315416929, 59379253693, 233381),
   (459016, 631278927, 133008010, 580)]

/--
The 35 exact integer rows for the order-`5 / 2` outer-component bounds and their rounded
budgets.
-/
def outerOrderFiveHalvesBounds : List OuterBoundRow :=
  [(7266522, 27, 1426, 1),
   (1241454497, 1, 821, 1),
   (1208324400, 1, 1392, 1),
   (1152630107, 1, 1765, 1),
   (1126190783, 1, 3334, 1),
   (1096246679, 1, 5753, 1),
   (1058983690, 1, 10303, 1),
   (967816560, 1, 18815, 1),
   (867471653, 1, 2089, 1),
   (603785822, 1, 11427, 1),
   (32188902, 16, 16011, 1),
   (1308239, 14362, 24579, 1),
   (386321, 7065761, 1054524, 6),
   (373849, 17914115, 2503735, 14),
   (377891, 260216687, 37159538, 197),
   (385136, 3305952377, 490372043, 2547),
   (395013, 38054077523, 5937779759, 30064),
   (405835, 352112119115, 57993623042, 285799),
   (419505, 3006707964277, 529135146833, 2522662),
   (432001, 19352692647427, 3611707956032, 16720799),
   (445139, 14498518468563, 2872865686933, 12907719),
   (457321, 28197429960534, 5897287451435, 25790569),
   (525975, 57148020076132, 15810035599715, 60116961),
   (518733, 69886316496332, 18805329967080, 72504766),
   (515168, 69366993102523, 18409874222209, 71471327),
   (512357, 62684551010344, 16455348517458, 64233828),
   (509770, 53862830801099, 13997130877252, 54915393),
   (507320, 44981355032435, 11577030541299, 45639918),
   (504951, 36911787941323, 9411611503675, 37277308),
   (502604, 29975466544992, 7572139370727, 30131606),
   (503256, 50573740961589, 12808689167903, 50903176),
   (498048, 32438336646873, 8046407139897, 32311736),
   (492222, 20308616081603, 4920425453752, 19992702),
   (485810, 12358345921158, 2916712121993, 12007621),
   (433769, 15056954296612, 2833062492447, 13062511)]

/--
The seven exact rows for order-two inner components of the base region, recording integral
bounds and rounded weighted budgets.
-/
def innerBaseOrderTwoBounds : List InnerBoundRow :=
  [(25777, 1), (1511410893, 14), (18120016651, 161),
   (903601038105, 8027), (425243194887, 3778),
   (4871216699917, 43272), (23946432, 1)]

/--
The ten exact rows for order-`5 / 2` inner components of the base region, recording integral
bounds and rounded weighted budgets.
-/
def innerBaseOrderFiveHalvesBounds : List InnerBoundRow :=
  [(1, 1), (3229104, 1), (29825526, 1), (77797373079, 692),
   (131978724894, 1173), (292684783730, 2600), (5548294545493, 49286),
   (30283518217418, 269010), (12009121688668, 106678), (686922192553, 6102)]

/--
The eleven exact rows for order-two inner components of the enlarged region, recording integral
bounds and rounded weighted budgets.
-/
def innerEnlargedOrderTwoBounds : List InnerBoundRow :=
  [(467789, 1), (381747797, 383), (386210860, 387),
   (99885644276, 99970), (247732013063, 247941), (381057139991, 381379),
   (266162792752, 266388), (337097314828, 337382), (34427294106, 34457),
   (36820947233, 36852), (18106118, 19)]

/--
The seventeen exact rows for order-`5 / 2` inner components of the enlarged region, recording
integral bounds and rounded weighted budgets.
-/
def innerEnlargedOrderFiveHalvesBounds : List InnerBoundRow :=
  [(2, 1), (107126908277, 107218), (1, 1), (61, 1), (137, 1),
   (177471603, 178), (327802576, 329), (50667881720, 50711),
   (143104919759, 143226), (1323952422879, 1325069), (697854132745, 698443),
   (4234127556194, 4237698), (11632061739670, 11641870),
   (3641610451935, 3644681), (6136054632765, 6141229),
   (3690866567521, 3693979), (737132501820, 737755)]

section
open Set
/--
The eleven multisets of exponents specifying the angular power-sum monomials in the trial
function. The empty multiset gives the constant monomial.
-/
def trialAngularSignature : Fin 11 → Multiset ℕ :=
  ![0, {2}, {3}, {4}, {5}, {6}, {2, 2}, {2, 3}, {2, 4}, {3, 3}, {2, 2, 2}]

/--
The integer coefficient table for the eleven degree-at-most-six radial polynomials. Actual
coefficients are obtained by dividing every entry by `10^10`.
-/
def trialCoefficientInteger : Fin 11 → Fin 7 → ℤ :=
  ![![10000000000, -264598476112, 834262268474, -3540575351215,
      5377491111325, 116705356572254, 121730820431102],
    ![71070047507, 7222861788586, -48747932657986, 290976672545723,
      -1136422724027134, -2058910631434711, 1375878942948547],
    ![6457252424873, -61201446212885, 28811649792090, -2058803084231281,
      9970156747759406, 38278849934023144, 34806920812932737],
    ![-16779263512274, 128033707910825, 169290603857215, 7359669931312727,
      -35090795379920588, -120997133235510923, -141527585901304670],
    ![28114161526671, -276375633435566, -690482702304933, -15120456749385986,
      47650276584638031, 372137534144492224, 2191913505882230103],
    ![-21150553032771, 379875924448266, -929062247569514, -13967219843969236,
      168416050564605519, -248058724714138769, -4161797957833172083],
    ![-19004216224617, 136674974139652, -183344485344333, 1114879023072977,
      1119583327002910, -47568062965320963, 2537631525616777],
    ![31171802814567, -97428388152051, 526635309233054, 1222047580792220,
      15846796161434448, -165623953461271580, -2358019996221938729],
    ![-53602626608739, 329397902441121, 1613505134333316, 13893066541430270,
      -126262434123562668, 239934302929501943, -1506861482386002243],
    ![27653330903418, -290549334488305, -1847330641348475, 28351866831729468,
      -61505472221886320, -424266003419714347, 2585236507449911535],
    ![12374547113901, -244168600145684, 1603694896120437, -21603130476787649,
      -30285492734943698, 381987976419637874, 3605516061450295448]]

/--
The radial polynomial attached to angular signature `s`, with seven coefficients taken from the
exact integer table and scaled by `10⁻¹⁰`.
-/
noncomputable def trialRadialPolynomial (s : Fin 11) : Polynomial ℚ :=
  Polynomial.ofFn 7 (fun d : Fin 7 => (trialCoefficientInteger s d : ℚ) / 10000000000)

/--
The product of power sums `∏ e ∈ σ, ∑ i, t i ^ e`, retaining multiplicities in `σ`. The empty
signature evaluates to `1`.
-/
def angularMonomial {R : Type*} [CommSemiring R] {d : ℕ}
    (σ : Multiset ℕ) (t : Fin d → R) : R :=
  (σ.map (fun e => ∑ i : Fin d, t i ^ e)).prod

end

/-- The rational mesh width `(2742997 / 2624989) / 98304` used to discretize fragment masses. -/
def trialMesh : ℚ := (2742997 / 2624989) / 98304

/-- The largest allowed fragment location, equal to `68225` mesh widths. -/
def trialLargestCap : ℚ := 68225 * trialMesh

/--
The physical fragment measure at the largest cap, obtained by scaling the fragment probability
law by `exp(γ)` times that cap.
-/
noncomputable def trialPhysicalMeasure : Measure (FiniteMeasure ℝ) :=
  ENNReal.ofReal (Real.exp Real.eulerMascheroniConstant * (trialLargestCap : ℝ)) •
    fragmentLaw (trialLargestCap : ℝ)

/--
The natural-number floor of a fragment measure's total mass divided by the mesh width. It
indexes the left-closed mesh cell containing the mass.
-/
noncomputable def trialCellIndex (X : FiniteMeasure ℝ) : ℕ :=
  ⌊(X.mass : ℝ) / (trialMesh : ℝ)⌋₊

/-- The rational midpoint `(j + 1 / 2) * trialMesh` of mesh cell `j`. -/
def trialCellMidpoint (j : ℕ) : ℚ := ((j : ℚ) + 1 / 2) * trialMesh

/--
The two-pole rational profile with weights `21 / 200` and `179 / 200`, used multiplicatively in
the trial function. Division follows the ambient field's totalized convention.
-/
def trialProfile {R : Type*} [Field R] (t : R) : R :=
  (21 / 200) / (1 + t / 100) + (179 / 200) / (1 + (907 / 5) * t)

/-- The exact rational trial-profile value at the midpoint of mesh cell `j`. -/
def trialProfileValue (j : ℕ) : ℚ :=
  trialProfile (trialCellMidpoint j)

def trialProfileNormalizer : ℚ :=
  ∑ j ∈ Finset.range 98264, trialProfileValue j ^ 2

/--
The indicator of the admissible 40-coordinate outer region: the summed cell index is at most
`98263`, and every fragment measure obeys the radial band's location cap.
-/
noncomputable def trialOuterMask (X : Fin 40 → FiniteMeasure ℝ) : ℝ := by
  classical
  exact
    let r : ℕ := ∑ i : Fin 40, trialCellIndex (X i)
    let cap : ℕ := if r ≤ 89196 then 68225 else if r ≤ 95598 then 49152 else 46580
    if r ≤ 98263 ∧
        ∀ i : Fin 40,
          (X i : Measure ℝ) (Set.Ioi ((cap : ℝ) * (trialMesh : ℝ))) = 0
    then 1 else 0

/--
The indicator of the base 39-coordinate region, with summed cell index at most `89524` and the
corresponding piecewise fragment-location cap.
-/
noncomputable def trialBaseMask (Y : Fin 39 → FiniteMeasure ℝ) : ℝ := by
  classical
  exact
    let r : ℕ := ∑ i : Fin 39, trialCellIndex (Y i)
    let cap : ℕ := if r ≤ 84930 then 68225 else if r ≤ 87194 then 44781 else 35265
    if r ≤ 89524 ∧
        ∀ i : Fin 39,
          (Y i : Measure ℝ) (Set.Ioi ((cap : ℝ) * (trialMesh : ℝ))) = 0
    then 1 else 0

/--
The indicator of the enlarged 39-coordinate region, with summed cell index at most `89914` and
its piecewise fragment-location cap.
-/
noncomputable def trialEnlargedMask (Y : Fin 39 → FiniteMeasure ℝ) : ℝ := by
  classical
  exact
    let r : ℕ := ∑ i : Fin 39, trialCellIndex (Y i)
    let cap : ℕ := if r ≤ 85161 then 68225 else if r ≤ 87249 then 44976 else 35419
    if r ≤ 89914 ∧
        ∀ i : Fin 39,
          (Y i : Measure ℝ) (Set.Ioi ((cap : ℝ) * (trialMesh : ℝ))) = 0
    then 1 else 0

/--
The indicator of the full 39-coordinate region: summed cell index at most `98263` and no
fragment mass above the largest cap.
-/
noncomputable def trialFullMask (Y : Fin 39 → FiniteMeasure ℝ) : ℝ := by
  classical
  exact
    if (∑ i : Fin 39, trialCellIndex (Y i)) ≤ 98263 ∧
        ∀ i : Fin 39,
          (Y i : Measure ℝ) (Set.Ioi (trialLargestCap : ℝ)) = 0
    then 1 else 0

/--
The masked trial expression: a product of one-variable profiles times eleven
radial-polynomial/angular-monomial terms. The radial polynomials are evaluated at `(∑ i, t i) -
9 / 10`, while the mask depends on the original fragment measures `X`.
-/
noncomputable def trialCore
    (X : Fin 40 → FiniteMeasure ℝ) (t : Fin 40 → ℝ) : ℝ :=
  trialOuterMask X * (∏ i : Fin 40, trialProfile (t i)) *
    ∑ s : Fin 11,
      (trialRadialPolynomial s).eval₂ (Rat.castHom ℝ) ((∑ i, t i) - 9 / 10) *
        angularMonomial (trialAngularSignature s) t

noncomputable def trialStepFunction (X : Fin 40 → FiniteMeasure ℝ) : ℝ :=
  trialCore X (fun i => (trialCellMidpoint (trialCellIndex (X i)) : ℝ))

noncomputable def trialMarginal
    (i : Fin 40) (Y : Fin 39 → FiniteMeasure ℝ) : ℝ :=
  ∫ X : FiniteMeasure ℝ, trialStepFunction (i.insertNth X Y) ∂trialPhysicalMeasure

/--
The fortieth power of the mesh-weighted one-dimensional squared-profile sum, used to normalize
all trial integrals.
-/
noncomputable def trialPhysicalNormalizer : ℝ :=
  ((trialMesh : ℝ) * (trialProfileNormalizer : ℝ)) ^ 40

/--
The normalized squared `L²` integral of the 40-coordinate step trial function under the product
physical measure.
-/
noncomputable def trialIH : ℝ :=
  (∫ X : Fin 40 → FiniteMeasure ℝ, trialStepFunction X ^ 2
    ∂Measure.pi (fun _ : Fin 40 => trialPhysicalMeasure)) / trialPhysicalNormalizer

/--
The normalized sum of squared coordinate marginals integrated over the base 39-coordinate
region.
-/
noncomputable def trialJ0 : ℝ :=
  (∑ i : Fin 40,
    ∫ Y : Fin 39 → FiniteMeasure ℝ,
      trialBaseMask Y * trialMarginal i Y ^ 2
      ∂Measure.pi (fun _ : Fin 39 => trialPhysicalMeasure)) / trialPhysicalNormalizer

/--
The normalized sum of squared coordinate marginals on the enlarged region outside the base
region.
-/
noncomputable def trialJPlus : ℝ :=
  (∑ i : Fin 40,
    ∫ Y : Fin 39 → FiniteMeasure ℝ,
      trialEnlargedMask Y * (1 - trialBaseMask Y) * trialMarginal i Y ^ 2
      ∂Measure.pi (fun _ : Fin 39 => trialPhysicalMeasure)) / trialPhysicalNormalizer

/--
The normalized sum of squared coordinate marginals on the full region outside the enlarged
region.
-/
noncomputable def trialJTail : ℝ :=
  (∑ i : Fin 40,
    ∫ Y : Fin 39 → FiniteMeasure ℝ,
      trialFullMask Y * (1 - trialEnlargedMask Y) * trialMarginal i Y ^ 2
      ∂Measure.pi (fun _ : Fin 39 => trialPhysicalMeasure)) / trialPhysicalNormalizer

/--
The signed trial marginal functional combining the base, enlargement, and tail contributions
with the specified exact rational coefficients.
-/
noncomputable def trialJLambdaH : ℝ :=
  trialJ0 +
    ((2479900401 / 2500000000 : ℝ) + (-843183 / 1000000000 : ℝ)) * trialJPlus +
      (-843183 / 1000000000 : ℝ) * trialJTail

/--
Exact rational geometry for one source row: its order, radial band, activation level, outer and
inner core radii, and their thresholds.
-/
structure PhysicalSourceRowData where
  /--
  Dense-divisibility order required by this source row; the canonical rows use orders `1`, `2`,
  and `3`.
  -/
  order : ℕ
  /--
  Exclusive lower endpoint of the row's modulus-size band in logarithmic exponent coordinates.
  -/
  lowerBand : ℚ
  /--
  Inclusive upper endpoint of the row's modulus-size band in logarithmic exponent coordinates.
  -/
  upperBand : ℚ
  /--
  Prime-factor exponent cutoff `ξ`, also determining the dense-divisibility scale `R^ξ`;
  activation requires an exponent strictly greater than `ξ`.
  -/
  activation : ℚ
  /--
  Outer total-mass cutoff at or below which no ownership constraint is needed; canonically the
  lower band endpoint minus the inner support radius.
  -/
  outerCore : ℚ
  /--
  Inner total-mass cutoff at or below which no ownership constraint is needed; canonically the
  lower band endpoint minus the outer support radius.
  -/
  innerCore : ℚ
  /--
  Upper budget for outer-owned mass; canonical rows add their order-dependent increment to
  `outerCore`.
  -/
  outerThreshold : ℚ
  /--
  Upper budget for inner-owned mass; canonical rows add their order-dependent increment to
  `innerCore`.
  -/
  innerThreshold : ℚ

/-- The fixed rational density parameter `262499 / 1000000` used in source-row geometry. -/
def physicalSourceRho : ℚ := 262499 / 1000000

/-- The outer radial cutoff, equal to `98304` mesh widths. -/
def physicalSourceOuterRadius : ℚ := 98304 * trialMesh

/-- The inner radial cutoff for the base (`ν = 0`) or enlarged (`ν = 1`) source family. -/
def physicalSourceInnerRadius (ν : Fin 2) : ℚ :=
  if ν = 0 then 2 - 3 / 1000 - physicalSourceOuterRadius else 2510000 / 2624989

/--
The positive radial advance `10⁻⁷ / physicalSourceRho` separating source thresholds from the
cutoff radii.
-/
def physicalSourceAdvance : ℚ := (1 / 10000000) / physicalSourceRho

/--
Assign source rows to orders one, two, and three using the index bands below `12`, from `12` to
`23`, and from `24` onward.
-/
def physicalSourceOrder (t : ℕ) : ℕ :=
  if t < 12 then 1 else if t < 24 then 2 else 3

/--
The rational intercept and slope governing the source-row activation inequality, selected by
source family and row order.
-/
def physicalSourceAffine (ν : Fin 2) (t : ℕ) : ℚ × ℚ :=
  let σ : ℚ :=
    if ν = 0 then 100001 / 1000000 else 1 / 2 - 40481 / 100000 + 1 / 10000000000
  if physicalSourceOrder t = 1 then ((1 - 5 * σ) / 15, 18 / 5)
  else if physicalSourceOrder t = 2 then ((1 - 4 * σ) / 16, 7 / 2)
  else if ν = 0 then (3 / 80, 3) else ((1 - 2 * σ) / 20, 16 / 5)

/--
The recursively advanced rational source parameter, starting at zero and capped at the
family-specific terminal value. Each step uses the row's affine bound with explicit safety
margins.
-/
def physicalSourceOmegaPrefix (ν : Fin 2) : ℕ → ℚ :=
  Nat.rec 0 (fun t previous =>
    let Ω : ℚ := if ν = 0 then 12499 / 1000000 else 253 / 20000
    let ε : ℚ := if ν = 0 then 1 / 1000000 else 1 / 10000000
    let E : ℚ := physicalSourceRho *
      (physicalSourceOuterRadius + physicalSourceInnerRadius ν) - 1 / 2
    let cs := physicalSourceAffine ν t
    if previous = Ω then Ω else
      min Ω ((cs.1 - ε - E + 2 * previous - 1 / 10000000) / cs.2))

/--
Construct the exact row geometry from two successive source parameters. Orders one and two use
the activation directly as the threshold increment; order three uses the half-adjusted
increment.
-/
def physicalSourceRow (ν : Fin 2) (t : ℕ) : PhysicalSourceRowData :=
  let cs := physicalSourceAffine ν t
  let ε : ℚ := if ν = 0 then 1 / 1000000 else 1 / 10000000
  let B : ℚ := (1 / 2 + 2 * physicalSourceOmegaPrefix ν t) / physicalSourceRho
  let Bplus : ℚ :=
    (1 / 2 + 2 * physicalSourceOmegaPrefix ν (t + 1)) / physicalSourceRho
  let ξ : ℚ :=
    (cs.1 - cs.2 * physicalSourceOmegaPrefix ν (t + 1) - ε) / physicalSourceRho
  let a : ℚ := B - physicalSourceInnerRadius ν
  let b : ℚ := B - physicalSourceOuterRadius
  let η : ℚ := if physicalSourceOrder t ≤ 2 then ξ else
    (ξ + physicalSourceOuterRadius + physicalSourceInnerRadius ν - B) / 2
  { order := physicalSourceOrder t
    lowerBand := B
    upperBand := Bplus
    activation := ξ
    outerCore := a
    innerCore := b
    outerThreshold := a + η
    innerThreshold := b + η }

/-- In both TeX documents, `cap` is the group cap `z_G` and `split` is `p_G`. -/
structure PhysicalSourceGroupData where
  /--
  Number of fragment-measure coordinates; canonical outer and inner groups use `40` and `39`.
  -/
  dimension : ℕ
  /--
  Effective covering order `μ_G`; canonical groups use `2` or `5 / 2`, distinct from a row's
  dense-divisibility order.
  -/
  order : ℚ
  /--
  Lower fragment-size endpoint of the low-fragment partition; canonical groups choose it no
  larger than their assigned rows' activation levels.
  -/
  activation : ℚ
  /--
  Budget `T_G` exceeded in the covering test `tailMass + (order - 1) * p > T_G`, where the tail
  includes fragments of size at least `p`.
  -/
  threshold : ℚ
  /--
  Exclusive lower endpoint of the group's total-fragment-mass range, before component-specific
  clipping and mesh-cell enlargement.
  -/
  lowerRadius : ℚ
  /--
  Inclusive upper endpoint of the group's total-fragment-mass range, used to bound its radial
  cell indices.
  -/
  upperRadius : ℚ
  /-- The group cap `z_G` on individual fragment sizes, not on their total mass. -/
  cap : ℚ
  /--
  The splitting point `p_G` ending the low-fragment partition; the triple-count component counts
  fragments strictly above it.
  -/
  split : ℚ

/--
The six covering groups: two outer groups, two base inner groups, and two enlarged inner groups,
each pair having orders `2` and `5 / 2`.
-/
def physicalSourceGroup (g : Fin 6) : PhysicalSourceGroupData :=
  let S := physicalSourceOuterRadius
  let T0 := physicalSourceInnerRadius 0
  let T1 := physicalSourceInnerRadius 1
  let e := physicalSourceAdvance
  ![{ dimension := 40, order := 2,
      activation := (physicalSourceRow 0 23).activation, threshold := S + e,
      lowerRadius := (physicalSourceRow 1 0).outerCore,
      upperRadius := (physicalSourceRow 1 24).outerCore,
      cap := 49152 * trialMesh, split := 24576 * trialMesh },
    { dimension := 40, order := 5 / 2,
      activation := (physicalSourceRow 1 38).activation, threshold := S + e / 2,
      lowerRadius := (physicalSourceRow 1 24).outerCore,
      upperRadius := 98303 * trialMesh,
      cap := 46580 * trialMesh, split := 19660 * trialMesh },
    { dimension := 39, order := 2,
      activation := (physicalSourceRow 0 23).activation, threshold := T0 + e,
      lowerRadius := (physicalSourceRow 0 12).innerCore,
      upperRadius := (physicalSourceRow 0 24).innerCore,
      cap := 44781 * trialMesh, split := 22390 * trialMesh },
    { dimension := 39, order := 5 / 2,
      activation := (physicalSourceRow 0 27).activation, threshold := T0 + e / 2,
      lowerRadius := (physicalSourceRow 0 24).innerCore,
      upperRadius := 89563 * trialMesh,
      cap := 35265 * trialMesh, split := 17912 * trialMesh },
    { dimension := 39, order := 2,
      activation := (physicalSourceRow 1 23).activation, threshold := T1 + e,
      lowerRadius := (physicalSourceRow 1 12).innerCore,
      upperRadius := (physicalSourceRow 1 24).innerCore,
      cap := 44976 * trialMesh, split := 22488 * trialMesh },
    { dimension := 39, order := 5 / 2,
      activation := (physicalSourceRow 1 38).activation, threshold := T1 + e / 2,
      lowerRadius := (physicalSourceRow 1 24).innerCore,
      upperRadius := 89953 * trialMesh,
      cap := 35419 * trialMesh, split := 17990 * trialMesh }] g

/-- The finite set of source-family and row-index pairs assigned to covering group `g`. -/
def physicalSourceRows (g : Fin 6) : Finset (Fin 2 × ℕ) :=
  ![(Finset.range 24).biUnion (fun t => {(0, t), (1, t)}),
    (Finset.Icc 24 27).image (fun t => (0, t)) ∪
      (Finset.Icc 24 38).image (fun t => (1, t)),
    (Finset.Icc 12 23).image (fun t => (0, t)),
    (Finset.Icc 24 27).image (fun t => (0, t)),
    (Finset.Icc 12 23).image (fun t => (1, t)),
    (Finset.Icc 24 38).image (fun t => (1, t))] g

/--
The exact rational endpoints partitioning the low-fragment part of group `g`, from its
activation level to its splitting point.
-/
def physicalSourceLowBoundaries (g : Fin 6) : List ℚ :=
  let ξ := (physicalSourceGroup g).activation
  let p := (physicalSourceGroup g).split
  let a : ℕ → ℚ := fun j => (1 / 20) * (6 / 5) ^ j
  ![[ξ, (3 / 2) * ξ, a 0, a 4, a 5, a 6, a 7, a 8, (a 8 + a 9) / 2, a 9, p],
    [ξ, 2 * ξ, 4 * ξ, 8 * ξ, 16 * ξ, 32 * ξ, 64 * ξ, 128 * ξ,
      256 * ξ, 1 / 100, 3 / 200, 9 / 400, 27 / 800, a 0, a 1, a 2,
      a 3, a 4, a 5, a 6, a 7, (a 7 + p) / 2, p],
    [ξ, a 0, a 4, a 6, a 8, p],
    [ξ, 2 * ξ, 1 / 100, 27 / 800, a 3, a 5, a 6, p],
    [ξ, a 1, a 4, a 5, a 7, a 8, p],
    [ξ, 2 * ξ, 16 * ξ, 64 * ξ, 256 * ξ, 9 / 400,
      a 1, a 3, a 5, a 6, a 7, p]] g

/--
The rational subdivision fractions used to partition the rank component interval between its
lower endpoint and the group cap.
-/
def physicalSourceRankFractions (g : Fin 6) : List ℚ :=
  ![[0, 1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6, 1],
    [0, 1 / 16, 1 / 8, 3 / 16, 1 / 4, 5 / 16, 3 / 8,
      7 / 16, 1 / 2, 5 / 8, 3 / 4, 7 / 8, 1],
    [0, 1],
    [0, 1 / 2, 1],
    [0, 1 / 6, 1 / 2, 2 / 3, 1],
    [0, 1 / 8, 3 / 8, 1 / 2, 3 / 4, 1]] g

/-- The number of consecutive low-fragment intervals, one less than the boundary-list length. -/
def physicalSourceLowCount (g : Fin 6) : ℕ := (physicalSourceLowBoundaries g).length - 1

/--
The number of consecutive rank intervals, one less than the subdivision-fraction list length.
-/
def physicalSourceRankCount (g : Fin 6) : ℕ := (physicalSourceRankFractions g).length - 1

/--
The total number of covering components: all low intervals, all rank intervals, and one
triple-count component.
-/
def physicalSourceRowCount (g : Fin 6) : ℕ :=
  physicalSourceLowCount g + physicalSourceRankCount g + 1

/--
Encode a component as `0` for low-fragment, `1` for rank, or `2` for triple-count. Indices
beyond the declared row count also receive `2` and are rejected by the radial mask.
-/
def physicalSourceComponentKind (g : Fin 6) (j : ℕ) : ℕ :=
  if j < physicalSourceLowCount g then 0
  else if j < physicalSourceLowCount g + physicalSourceRankCount g then 1 else 2

/--
The rational endpoints of the chosen component: consecutive low boundaries, affinely rescaled
rank fractions, or the split-to-cap interval for the triple-count component.
-/
def physicalSourceComponentEndpoints (g : Fin 6) (j : ℕ) : ℚ × ℚ :=
  let G := physicalSourceGroup g
  if physicalSourceComponentKind g j = 0 then
    ((physicalSourceLowBoundaries g).getD j 0,
      (physicalSourceLowBoundaries g).getD (j + 1) 0)
  else if physicalSourceComponentKind g j = 1 then
    let k := j - physicalSourceLowCount g
    let q0 := G.threshold / (G.order + 1)
    (q0 + (physicalSourceRankFractions g).getD k 0 * (G.cap - q0),
      q0 + (physicalSourceRankFractions g).getD (k + 1) 0 * (G.cap - q0))
  else (G.split, G.cap)

/--
The exponential tilt for a low-fragment component, rounded upward from `7 / upperEndpoint` or `9
/ upperEndpoint`, with the specified exceptional value `120`. Non-low components have zero tilt.
-/
def physicalSourceTheta (g : Fin 6) (j : ℕ) : ℚ :=
  if physicalSourceComponentKind g j = 0 then
    if g = 0 ∧ j = 2 then 120 else
      let numerator : ℚ := if g = 5 then 9 else 7
      ((⌈numerator / (physicalSourceComponentEndpoints g j).2⌉ : ℤ) : ℚ)
  else 0

/--
The lower radial clipping level obtained from eligible source rows activated below `u`,
including rows absorbed from the preceding group where prescribed. It is absent when no row is
eligible; otherwise it is the smallest row bound, truncated below by the group's lower radius.
-/
def physicalSourceLowClipping (g : Fin 6) (u : ℚ) : Option ℚ :=
  let G := physicalSourceGroup g
  let absorbed : Finset (Fin 2 × ℕ) :=
    if g = 1 then physicalSourceRows 0
    else if g = 3 then physicalSourceRows 2
    else if g = 5 then physicalSourceRows 4 else ∅
  let eligible := (physicalSourceRows g ∪ absorbed).filter
    (fun row => (physicalSourceRow row.1 row.2).activation < u)
  let values : Finset ℚ := eligible.image (fun row =>
    let R := physicalSourceRow row.1 row.2
    let core := if g.val < 2 then R.outerCore else R.innerCore
    let threshold := if g.val < 2 then R.outerThreshold else R.innerThreshold
    let μ : ℚ := if R.order ≤ 2 then 2 else 5 / 2
    max core (threshold - (μ - 1) * u))
  if hvalues : values.Nonempty then some (max G.lowerRadius (values.min' hvalues)) else none

/--
The fragment-cap index aligned with radial cell sum `r`, using the outer, base-inner, or
enlarged-inner piecewise schedule according to the group.
-/
def physicalSourceAlignedCapIndex (g : Fin 6) (r : ℕ) : ℕ :=
  if g.val < 2 then
    if r ≤ 89196 then 68225 else if r ≤ 95598 then 49152 else 46580
  else if g.val < 4 then
    if r ≤ 84930 then 68225 else if r ≤ 87194 then 44781 else 35265
  else
    if r ≤ 85161 then 68225 else if r ≤ 87249 then 44976 else 35419

/--
The indicator that a component exists and its radial cell sum satisfies the global cutoff and
clipped group bounds. The lower bound includes the dimension-dependent mesh-cell correction.
-/
def physicalSourceRadialMask (g : Fin 6) (j d r : ℕ) : ℝ :=
  let G := physicalSourceGroup g
  let top : ℕ := if g.val < 2 then 98263 else if g.val < 4 then 89524 else 89914
  let lower : Option ℚ := if physicalSourceComponentKind g j = 0 then
    physicalSourceLowClipping g (physicalSourceComponentEndpoints g j).2
    else some G.lowerRadius
  match lower with
  | none => 0
  | some c =>
    if j < physicalSourceRowCount g ∧ r ≤ top ∧
        (⌊c / trialMesh⌋ : ℤ) - (d : ℤ) + 1 ≤ (r : ℤ) ∧
        (r : ℤ) ≤ (⌊G.upperRadius / trialMesh⌋ : ℤ)
    then 1 else 0

/--
Sum the coordinate fragment measures on positive locations and divide their density by the
location. For weighted fragment atoms `u • δ_u`, this recovers the corresponding counting
measure.
-/
noncomputable def physicalSourceCountMeasure {d : ℕ}
    (X : Fin d → FiniteMeasure ℝ) : Measure ℝ :=
  ((∑ i : Fin d, (X i : Measure ℝ)).restrict (Set.Ioi (0 : ℝ))).withDensity
    (fun t : ℝ => ENNReal.ofReal t⁻¹)

/--
The nonnegative covering weight for one physical component, subject to radial and fragment-cap
masks. Low components use an exponentially tilted count, rank components integrate a rank
condition, and the final component counts triples above the splitting point.
-/
noncomputable def physicalSourceCover (g : Fin 6) (j : ℕ) {d : ℕ}
    (X : Fin d → FiniteMeasure ℝ) : ℝ := by
  classical
  exact
    let P := physicalSourceGroup g
    let e := physicalSourceComponentEndpoints g j
    let r : ℕ := ∑ i : Fin d, trialCellIndex (X i)
    let N := physicalSourceCountMeasure X
    let χ := physicalSourceRadialMask g j d r
    let cap : ℝ := (physicalSourceAlignedCapIndex g r : ℝ) * (trialMesh : ℝ)
    if ∀ i : Fin d, (X i : Measure ℝ) (Set.Ioi cap) = 0 then
      χ *
        (if physicalSourceComponentKind g j = 0 then
          N.real (Set.Ioc (e.1 : ℝ) (e.2 : ℝ)) *
            Real.exp ((physicalSourceTheta g j : ℝ) *
              ((∑ i : Fin d, ((X i).mass : ℝ)) +
                ((P.order : ℝ) - 1) * (e.2 : ℝ) - (P.threshold : ℝ) -
                ∑ i : Fin d, (X i : Measure ℝ).real (Set.Ioc (0 : ℝ) (e.1 : ℝ))))
        else if physicalSourceComponentKind g j = 1 then
          ∫ q : ℝ in Set.Ioc (e.1 : ℝ) (e.2 : ℝ),
            (if N (Set.Ioi q) = 0 ∧
                (2 : ℝ) ≤ N.real (Set.Ioc (((P.threshold : ℝ) - q) / (P.order : ℝ)) q)
              then (1 : ℝ) else 0) ∂N
        else
          (Nat.choose ⌊N.real (Set.Ioi (P.split : ℝ))⌋₊ 3 : ℝ))
    else 0

/--
The absolute-coefficient envelope on a 39-coordinate face: `1` on the base region, the larger
enlargement/tail coefficient magnitude on the enlarged region, and the tail coefficient
magnitude elsewhere.
-/
noncomputable def physicalSourceFaceWeight
    (Y : Fin 39 → FiniteMeasure ℝ) : ℝ := by
  classical
  exact
    if trialBaseMask Y = 1 then 1
    else if trialEnlargedMask Y = 1 then
      max |(2479900401 / 2500000000 : ℝ) + (-843183 / 1000000000 : ℝ)|
        |(-843183 / 1000000000 : ℝ)|
    else |(-843183 / 1000000000 : ℝ)|

/--
The normalized outer-component integral of the squared trial function, summed over all 40 faces
and weighted by the covering function and face-coefficient envelope.
-/
noncomputable def physicalSourceOuterRoot (g : Fin 6) (j : ℕ) : ℝ :=
  (∑ i : Fin 40,
    ∫ X : Fin 40 → FiniteMeasure ℝ,
      physicalSourceCover g j X * physicalSourceFaceWeight (i.removeNth X) *
        trialStepFunction X ^ 2
      ∂Measure.pi (fun _ : Fin 40 => trialPhysicalMeasure)) / trialPhysicalNormalizer

/--
The normalized outer-component integral of the squared face marginal, summed over all 40 faces
with the covering function and face-coefficient envelope.
-/
noncomputable def physicalSourceOuterFace (g : Fin 6) (j : ℕ) : ℝ :=
  (∑ i : Fin 40,
    ∫ X : Fin 40 → FiniteMeasure ℝ,
      physicalSourceCover g j X * physicalSourceFaceWeight (i.removeNth X) *
        trialMarginal i (i.removeNth X) ^ 2
      ∂Measure.pi (fun _ : Fin 40 => trialPhysicalMeasure)) / trialPhysicalNormalizer

/--
The normalized covering-weighted squared-marginal integral for an inner component, summed over
coordinates. Groups below `4` use the base mask; the remaining groups use the enlarged mask.
-/
noncomputable def physicalSourceInnerMass (g : Fin 6) (j : ℕ) : ℝ :=
  (∑ i : Fin 40,
    ∫ Y : Fin 39 → FiniteMeasure ℝ,
      (if g.val < 4 then trialBaseMask Y else trialEnlargedMask Y) *
        physicalSourceCover g j Y * trialMarginal i Y ^ 2
      ∂Measure.pi (fun _ : Fin 39 => trialPhysicalMeasure)) / trialPhysicalNormalizer

end PrimeGap186

axiom PrimeGap186.physical_integral_bounds :
  (∀ j : Fin PrimeGap186.outerOrderTwoBounds.length,
    PrimeGap186.physicalSourceOuterRoot 0 j.val /
        ((23685317816 : ℝ) / (10 : ℝ) ^ 24) ≤
      ((PrimeGap186.outerOrderTwoBounds.get j).2.1 : ℝ) / (10 : ℝ) ^ 18 ∧
    PrimeGap186.physicalSourceOuterFace 0 j.val /
        ((23685317816 : ℝ) / (10 : ℝ) ^ 24) ≤
      ((PrimeGap186.outerOrderTwoBounds.get j).2.2.1 : ℝ) / (10 : ℝ) ^ 18) ∧
  (∀ j : Fin PrimeGap186.outerOrderFiveHalvesBounds.length,
    PrimeGap186.physicalSourceOuterRoot 1 j.val /
        ((23685317816 : ℝ) / (10 : ℝ) ^ 24) ≤
      ((PrimeGap186.outerOrderFiveHalvesBounds.get j).2.1 : ℝ) / (10 : ℝ) ^ 18 ∧
    PrimeGap186.physicalSourceOuterFace 1 j.val /
        ((23685317816 : ℝ) / (10 : ℝ) ^ 24) ≤
      ((PrimeGap186.outerOrderFiveHalvesBounds.get j).2.2.1 : ℝ) / (10 : ℝ) ^ 18) ∧
  (∀ j : Fin PrimeGap186.innerBaseOrderTwoBounds.length,
    PrimeGap186.physicalSourceInnerMass 2 j.val /
        ((23685317816 : ℝ) / (10 : ℝ) ^ 24) ≤
      ((PrimeGap186.innerBaseOrderTwoBounds.get j).1 : ℝ) / (10 : ℝ) ^ 18) ∧
  (∀ j : Fin PrimeGap186.innerBaseOrderFiveHalvesBounds.length,
    PrimeGap186.physicalSourceInnerMass 3 j.val /
        ((23685317816 : ℝ) / (10 : ℝ) ^ 24) ≤
      ((PrimeGap186.innerBaseOrderFiveHalvesBounds.get j).1 : ℝ) / (10 : ℝ) ^ 18) ∧
  (∀ j : Fin PrimeGap186.innerEnlargedOrderTwoBounds.length,
    PrimeGap186.physicalSourceInnerMass 4 j.val /
        ((23685317816 : ℝ) / (10 : ℝ) ^ 24) ≤
      ((PrimeGap186.innerEnlargedOrderTwoBounds.get j).1 : ℝ) / (10 : ℝ) ^ 18) ∧
  (∀ j : Fin PrimeGap186.innerEnlargedOrderFiveHalvesBounds.length,
    PrimeGap186.physicalSourceInnerMass 5 j.val /
        ((23685317816 : ℝ) / (10 : ℝ) ^ 24) ≤
      ((PrimeGap186.innerEnlargedOrderFiveHalvesBounds.get j).1 : ℝ) / (10 : ℝ) ^ 18) ∧
  (23685317816 : ℝ) / (10 : ℝ) ^ 24 ≤ PrimeGap186.trialIH ∧
  PrimeGap186.trialIH ≤ (23685317890 : ℝ) / (10 : ℝ) ^ 24 ∧
  (90248755123 : ℝ) / (10 : ℝ) ^ 24 ≤ PrimeGap186.trialJLambdaH

end

namespace PrimeGap186

/-- The forty shifts of diameter 186 in Equation (1.3) of the main paper, used in the proof of
Corollary 1.2. -/
def admissibleTuple : Finset ℕ :=
  {0, 2, 6, 12, 20, 26, 30, 32, 36, 42, 48, 50, 56, 60, 68, 72, 78, 86, 90, 92,
    98, 102, 110, 116, 120, 126, 132, 138, 140, 146, 152, 156, 158, 162,
    168, 170, 176, 180, 182, 186}

/-- The extended-real liminf of consecutive prime gaps; `Nat.nth Nat.Prime 0 = 2`. -/
noncomputable def primeGapLiminf : EReal :=
  Filter.liminf
    (fun n : ℕ =>
      (Nat.nth Nat.Prime (n + 1) : EReal) - (Nat.nth Nat.Prime n : EReal))
    Filter.atTop

/-- Theorem 1.1 of the main paper: every admissible integer 40-tuple has two-prime translates. -/
theorem dhl_40_2
    (H : Finset ℤ) (hcard : H.card = 40)
    (hadm : ∀ p : ℕ, p.Prime → ∃ a : ZMod p, ∀ h ∈ H, (h : ZMod p) ≠ a) :
    Set.Infinite {n : ℤ | 2 ≤ (H.filter (fun h => (n + h).toNat.Prime)).card} := by
  sorry

/-- Supporting conclusion in the proof of Corollary 1.2 for the explicit admissible tuple. -/
theorem infinite_two_prime_translates_admissibleTuple :
    Set.Infinite {n : ℕ |
      2 ≤ (admissibleTuple.filter (fun h => (n + h).Prime)).card} := by
  sorry

/-- Corollary 1.2 of the main paper: the lower limit of consecutive prime gaps is at most 186. -/
theorem primeGapLiminf_le_186 : primeGapLiminf ≤ (186 : EReal) := by
  sorry

end PrimeGap186
