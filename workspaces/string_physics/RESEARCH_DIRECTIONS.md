# Research Directions — SM from F-theory LVS via GL=12/D₆

**Scope:** 333 candidates (124 hypotheses + 209 cross-domain gaps), 7 domains, 21 cross-domain pairs  
**Object:** GL=12/D₆ polytope · h¹¹=17 · h²¹=20 · KS index 37 · 39/39 elliptic fibrations contain SM gauge factors  
**Open question:** is 39/39 structural or Kodaira artifact? What discriminates beyond gauge algebra containment?

---

## How to read this

Skeletal. Each direction is a cluster of underlying raw candidates (H = hypothesis, X = cross-domain gap). Terse because the reader can connect dots. Organized into tiers by physics value, then lettered within tiers. Dependencies noted where they exist.

```
ID. TITLE
    Cluster:    raw candidate IDs absorbed
    Bet:        what you're betting is true
    Kill shot:  what falsifies it
    Entry:      first CyTools computation
    Depends on: other directions that must work first
```

---

## TIER 1 — The D₆ Yukawa Program

Everything in Tier 1 is one interconnected program. If D₆ acts nontrivially on generations, the whole tier opens. If it acts trivially, most of it collapses.

### 1A. D₆ Irrep Decomposition on 3 Generations

    Cluster:    H2, H5, H7, H8, H21, H23, H25, H103
    Bet:        D₆ acts as E₁ ⊕ A₁ (2+1 split) on generations, producing texture zeros in 3×3 Yukawa matrices
    Kill shot:  decomposition is 3×A₁ (trivial) — no texture zeros, no predictions
    Entry:      Polytope(37).automorphisms() → action on (2,1)-form basis → Clebsch-Gordan decomposition
    Depends on: nothing — this is the gate

**Sub-targets if 1A works:**
- θ₁₂ ≈ 35° from D₆ texture (H8)
- θ₁₃ ≈ 8.5° from seesaw + D₆ (H2, H23)
- Cabibbo angle from D₆ up-quark texture zeros (H21, H103)
- Majorana CP phases from D₆-invariant period phases (H5)
- δ_CP discrimination between normal/inverted hierarchy (H7)
- Δm²₃₂ and Δm²₂₁ ratios from seesaw + RG running (H25)
- Normal hierarchy preference from D₆ constraints (H23)

### 1B. Monodromy Survival of D₆

    Cluster:    H4, H6, X5, X6
    Bet:        Picard-Fuchs monodromy commutes with enough D₆ generators that flavor symmetry survives to low energy
    Kill shot:  monodromy-stable subspace dim < 3, or only trivial D₆ elements survive
    Entry:      compute GKZ system from GL=12 lattice points → extract monodromy matrices at singular loci
    Depends on: 1A (must know D₆ acts nontrivially first)

**What this determines:**
- which of 26 Yukawa couplings actually contribute at low energy
- whether D₆ breaks to subgroup via monodromy → smaller texture
- GUT scale prediction from monodromy + D₆ combined (H6)

### 1C. D₆ Breaking via D-Branes

    Cluster:    H11, H22, H72
    Bet:        D-brane wrapping on non-invariant divisors provides controlled D₆ → subgroup breaking with computable selection rules
    Kill shot:  all tadpole-consistent configs either preserve full D₆ or break completely (no intermediate subgroups)
    Entry:      enumerate divisor D₆-orbits → tadpole consistency → classify surviving symmetry
    Depends on: 1A, 1B

**Sub-structure:**
- fractional D-branes at orbifold singularities: H²(D₆, U(1)) ≅ Z₂ discrete torsion (H11)
- non-invariant divisor wrappings introduce Yukawa selection rules (H22)
- non-Abelian → Abelian decomposition controls mass hierarchy (H72)

### 1D. D₆-Invariant Rational Curves → Yukawa Selection

    Cluster:    H20, H106
    Bet:        fixed-point loci of D₆ on rational curve families generate vanishing pattern of 26 couplings via Gromov-Witten selection rules
    Kill shot:  ≤8 of 26 couplings show geometric selection rules matching fermion hierarchy
    Entry:      D₆ action on Mori cone → equivariant localization on M̄₀,₃(X,β)
    Depends on: 1A

**What this adds:**
- worldsheet instanton contributions to Yukawa organized by D₆ irreps (H106)
- GW vanishing: ≥60% of degree ≤5 curve classes should vanish under D₆ selection (H106)
- rational curve fixed loci directly generate texture zeros (H20)

### 1E. Equivariant Vector Fields and Yukawa Deformations

    Cluster:    H104
    Bet:        D₆-equivariant holomorphic vector fields on Kähler moduli space determine which 3 of 26 Yukawa couplings can be deformed, others are rigid
    Kill shot:  >6 couplings have nonzero Lie derivative along D₆-invariant fields
    Entry:      5D D₆-invariant Kähler subspace → Weil-Petersson connection → Lie derivatives of Yukawa entries
    Depends on: 1A

### 1F. Nakayama Divisibility Bounds on D₆ Sublattice

    Cluster:    H110
    Bet:        Nakayama index-theoretic bounds on D₆-invariant sublattice of H²(X,Z) predict ≥30% of 26 couplings arithmetically forbidden
    Kill shot:  fewer than 4 entries forbidden
    Entry:      restrict intersection form to 5D D₆-invariant Kähler sublattice → divisibility analysis
    Depends on: 1A

### 1G. Mirror Quintic Monodromy → Coupling Zeros

    Cluster:    H71
    Bet:        mirror quintic monodromy around conifold/LCS determines which D₆-invariant Yukawa couplings vanish on the original quintic
    Kill shot:  ≤8 of 26 coupling vanishing patterns predicted correctly
    Entry:      analytic continuation of mirror period vector → monodromy orbit → coupling map
    Depends on: 1B

---

## TIER 2 — GL=12 Infrastructure

Computations that must happen before Tier 1 can produce numbers. Less glamorous, essential.

### 2A. Orientifold Involution Census for LVS

    Cluster:    H9, H75, H77, H80, H81
    Bet:        a D₆-compatible orientifold of GL=12 admits Swiss-cheese volume structure required for LVS
    Kill shot:  no compatible involution has large 4-cycle + small blow-up hierarchy
    Entry:      CalabiYau.divisors() → invariant combinations → h±₁,₁ and h±₂,₁ for each involution
    Depends on: nothing

**Sub-structure:**
- O3 vs O7 projection comparison: different involutions → different flux landscapes (H77)
- O-plane fixed-point loci create singularities in period-integral domain (H80)
- orientifold reduces PF system rank via GKZ lattice automorphism (H81)
- σ-invariant flux superpotential reduces vacuum solution space (H75)
- all feed back to which moduli get stabilized → constrains Tier 1

### 2B. K3 Picard Lattice as Hypercharge Filter

    Cluster:    H10
    Bet:        K3 base Picard lattice invariants (rank, signature, discriminant) correlate with U(1)_Y embedding viability and Mordell-Weil rank
    Kill shot:  Spearman ρ < 0.3 across 39 fibrations
    Entry:      divisor_basis() → K3 base divisors → intersection form → Picard lattice for each of 39 fibrations
    Depends on: nothing

### 2C. Weierstrass Discriminant Rank-Defect → U(1)_Y Embedding

    Cluster:    H87
    Bet:        rank-defect of Weierstrass discriminant determines finite set of valid U(1)_Y embeddings via gl(n) completion of sl(3)⊕sl(2) Kodaira factors
    Kill shot:  >64% of 39 models admit arbitrary hypercharge embeddings
    Entry:      Weierstrass model for GL=12 fibrations → discriminant rank analysis
    Depends on: nothing

### 2D. Log Canonical Bundle → Chirality Filter

    Cluster:    H89
    Bet:        K_X(D) intersection invariants (D = singular Kodaira fiber divisor) predict chirality index and Yukawa rank without explicit flux computation
    Kill shot:  <40% sensitivity for identifying 3-generation + Yukawa rank ≥5 models
    Entry:      Kodaira fiber classification for GL=12 → log-discrepancy computation → intersection lattice
    Depends on: nothing

### 2E. Potential Barrier Heights from Kähler Geometry

    Cluster:    H88
    Bet:        χ=-6 polytopes with certain Kodaira types systematically produce sub-Planckian barriers in SM scalar directions, making tunneling timescales predictable
    Kill shot:  >50% of 39 polytopes have V_barrier > 10¹⁰ GeV
    Entry:      V_eff for each fibration from Kähler metric + superpotential
    Depends on: 2A (need orientifold + stabilization data)

### 2F. Reflexive Polygon Fiber → D₆ Decomposition

    Cluster:    H102
    Bet:        D₆ decomposes as G_fiber ⋊ G_base, where G_fiber from reflexive polygon encodes fiber-controlled Yukawa zeros
    Kill shot:  no reflexive polygon fiber of GL=12 maps to D₆ subgroup of order ≥3
    Entry:      enumerate toric fibrations of GL=12 → identify 2D fiber polygon → Aut(Δ₂) → map to Aut(Δ₄)
    Depends on: nothing

### 2G. Affine Chart Local Symmetry Enhancement

    Cluster:    H105
    Bet:        local affine chart automorphisms of GL=12 fan sometimes exceed D₆ → Yukawa couplings localized on enhanced-symmetry patches are suppressed below 10⁻³
    Kill shot:  all maximal cone automorphisms are exactly D₆ (no enhancement)
    Entry:      normal fan of GL=12 → maximal cones → Aut(σ) for each → intersection ∩_σ Aut(σ) = D₆
    Depends on: nothing

---

## TIER 3 — The Landscape Filter Stack

A pipeline of computational filters to narrow the search space before expensive phenomenological computation.

### 3A. Discrete Symmetry Census of χ=-6 Polytopes

    Cluster:    X7, X8, X10, X11, H122
    Bet:        automorphism group type/order correlates with SM viability across KS database, enabling pre-filter
    Kill shot:  SM-viable CY3s uniformly distributed across symmetry types
    Entry:      Polytope(id).automorphisms() across all χ=-6 polytopes
    Depends on: nothing

**Bonus question (H122):** Is GL=12 the unique maximal-symmetry χ=-6 polytope (D₆, order 12)? If yes, maximality itself is the distinguishing feature.

### 3B. Fano Sub-Geometry Gate

    Cluster:    X14, X17, H107, X34, X35, X58, X59
    Bet:        Fano divisors in elliptic CY3s are finitely classifiable (Mori-Mukai), and SM-viable fibrations cluster around specific Fano types
    Kill shot:  Fano types distribute uniformly across viable/non-viable
    Entry:      effective_cone() for 39 fibrations → identify Fano divisors → Mori-Mukai match
    Depends on: nothing

**Sub-structure:**
- Fano→quasi-Fano deformation: D₆ preserved iff singularities miss CY hypersurface (H107)
- anticanonical system |−K_X| automorphisms propagate to flavor symmetries (X34)
- Fano Picard lattice → Fano variety → moduli stabilization constraints (X59)

### 3C. CY4→CY3 Reduction Map

    Cluster:    X12, X15, X30, X32
    Bet:        F-theory CY4 data (Hodge numbers, discriminant loci, 7-brane configs) systematically determines which CY3 vacua inherit SM structure
    Kill shot:  chirality/generation count changes during geometric transition for GL=12 family
    Entry:      construct CY4 elliptic fibration over GL=12 base → Hodge numbers → discriminant locus
    Depends on: nothing

**Sub-structure:**
- CY4 F-theory 6D → CY3 4D: does chirality survive? (X12)
- CY4 discrete symmetries → 4D flavor structure via dimensional reduction (X30, X32)

### 3D. AdS Vacuum Stability as Joint Filter

    Cluster:    X13, X16
    Bet:        moduli-stabilization + AdS stability constraints jointly filter CY3s, and CY3s with SM gauge structure are over-represented in stable AdS minima
    Kill shot:  stable AdS minima show no SM gauge structure preference
    Entry:      1000 χ=-6 CY3s → Kähler cone geometry → flux superpotential → AdS stability eigenvalues
    Depends on: 2A (orientifold data)

### 3E. Characteristic Class Pre-Filter

    Cluster:    H101
    Bet:        c₁=0 + c₂ factorization patterns reduce slope-stable bundle parameter space by ≥30% before stability computation
    Kill shot:  reduction <30% or false-negative rate >15%
    Entry:      compute c₂ factorization for GL=12 resolved toric variety → bundle enumeration
    Depends on: nothing

### 3F. Flux Quantization as Dominant Landscape Reducer

    Cluster:    H95, H96
    Bet:        flux-quantization compatibility with CY topology (not gauge-sector structure) is the primary bottleneck, eliminating ~99.9% of configurations before moduli stabilization
    Kill shot:  >1% of flux configurations pass quantization compatibility
    Entry:      enumerate flux lattice for GL=12 → count compatible configurations → compare with stabilizable fraction
    Depends on: 2A

### 3G. SM Anomaly Constraints as Over-Determination

    Cluster:    H97, H91
    Bet:        anomaly cancellation + modular invariance over-constrain the SM chiral spectrum, with >60% of observed matter multiplicities determined by gauge-theoretic consistency alone
    Kill shot:  >100 topologically distinct fermion spectrum classes survive anomaly constraints
    Entry:      solve anomaly polynomial system for SU(3)×SU(2)×U(1) → count solutions → compare with observed spectrum
    Depends on: nothing

### 3H. Tangent Cone at Infinity → LVS Obstruction

    Cluster:    H90
    Bet:        tangent cone link topology of CY3 moduli space eliminates ≥30% of non-viable candidates by topological obstruction to Swiss-cheese completion
    Kill shot:  <10% eliminated
    Entry:      compute tangent cone link for GL=12 moduli space metric
    Depends on: 2A

---

## TIER 4 — Period/Moduli Deep Structure

Mathematical results about the Picard-Fuchs system, special geometry, and mirror symmetry that feed into Tiers 1-3 but require serious computation.

### 4A. DHT Conjecture as Vacuum Selector

    Cluster:    X9, X12
    Bet:        Tyurin degeneration structure of CY3s constrains moduli spaces accessible to chirality-preserving Yukawa couplings → new SM selection criterion
    Kill shot:  GL=12 doesn't admit Tyurin degeneration, or DHT-compatible CY3s show no SM preference
    Entry:      look for reducible anticanonical divisors in effective_cone() → Tyurin structure
    Depends on: nothing

### 4B. Weak Coupling Regime ↔ SM Selection Feedback

    Cluster:    X10, X13
    Bet:        g_s ≪ 1 selects for CY3s with special Hodge diamond properties; SM criteria restrict accessible g_s ranges → bidirectional constraint
    Kill shot:  no correlation between g_s and Hodge diamond features across explicit examples
    Entry:      enumerate CY3s with computed periods + known moduli stabilization → extract g_s → correlate with SM viability
    Depends on: 2A, 4A

### 4C. CY Arithmetic → Moduli Stabilization → Phenomenology Pathway

    Cluster:    X11, X14
    Bet:        period integrals + PF system structure + monodromy matrices → moduli-stabilization feasibility → gauge groups/matter/couplings: a computable causal chain
    Kill shot:  causal chain breaks at an intermediate step (e.g., stabilization feasibility doesn't correlate with phenomenological viability)
    Entry:      5000-10000 KS CYs: extract arithmetic invariants → stabilize where possible → measure phenomenological properties
    Depends on: 3A (symmetry census enables subset selection)

### 4D. SYZ Mirror as Period Computation Channel

    Cluster:    H29, H113
    Bet:        SYZ special Lagrangian fibration yields period integrals with ≥90% agreement to GKZ-derived periods, providing alternative computation path
    Kill shot:  ||Π^SYZ − Π^PF||²/||Π^PF||² > 0.20 for >30% of test points
    Entry:      construct SYZ fibration for GL=12/D₆ → fiber-by-fiber integration → compare with PF
    Depends on: 1B (need PF periods for comparison)

### 4E. Hessian of Flux Superpotential → Kähler Metric

    Cluster:    H28
    Bet:        ∂²W/∂z_i∂z_j at stabilization causally determines Kähler-moduli space metric via mirror map projection
    Kill shot:  Hessian rank ≤2 at stabilization (too degenerate)
    Entry:      compute W = G_a Π^a(z) at stabilization → Hessian → compare with Kähler metric → mirror map test
    Depends on: 1B, 2A

### 4F. Type IIA/IIB Prepotential Duality Check

    Cluster:    H30, H79
    Bet:        IIA prepotential F(X^a) from mirror PF periods constrains IIB Kähler moduli metric → stabilization landscape
    Kill shot:  ∂³F/∂X_a∂X_b∂X_c mismatch >10% with IIB triple intersections
    Entry:      solve PF system → extract A-periods → prepotential → compare with IIB intersection numbers
    Depends on: 1B

### 4G. Rigid CY (h²¹→0) Attractor Phenomenon

    Cluster:    H76, H100
    Bet:        rigidity drives PF rank reduction → residual modular-form constraint → attractor locus in stabilization landscape
    Kill shot:  period derivatives remain bounded as h²¹→0 (no attractor)
    Entry:      one-parameter family containing GL=12 → track PF system as h²¹ decreases
    Depends on: 1B

**Related:** rigid CY as boundary of non-Kähler moduli space (H100): does h¹¹=0 obstruct balanced-metric continuation via Dolbeault H¹(TX)?

### 4H. Relative Cohomology → Monodromy Without GKZ

    Cluster:    H84
    Bet:        H³(X, X_∞; Z) module structure determines complete monodromy + PF singularity structure without computing GKZ operators
    Kill shot:  <85% concordance with GKZ-derived exponents for GL=12/D₆
    Entry:      Bott-Safarevič generators → cycle transport → monodromy matrices → compare with GKZ
    Depends on: nothing (alternative to 1B)

### 4I. GKZ Inverse Functor: Operators → Polytope

    Cluster:    H85
    Bet:        principal part of GKZ operators uniquely determines the toric lattice polytope via computable inverse functor
    Kill shot:  reconstructed charge matrix differs in rank from true charge matrix
    Entry:      take known GKZ system → extract principal symbols → characteristic variety → reconstruct polytope → check
    Depends on: nothing

---

## TIER 5 — Moduli Stabilization Specifics

### 5A. F-term Uplift Microscopic Constraints

    Cluster:    H78
    Bet:        first-principles F-term backreaction (anti-D3 or gaugino condensation) rules out ≥30% of proposed uplift scenarios by Hessian instability
    Kill shot:  computed V_uplift matches naive parametric estimate within ±10%
    Entry:      GL=12 → F-term contribution to scalar potential → Hessian eigenvalues at stabilization
    Depends on: 2A

### 5B. Inflationary Constraints on Kähler Moduli

    Cluster:    H74
    Bet:        slow-roll surface (ε<0.01, |η|<0.01) intersects LVS locus in codimension-≥2 subvariety, leaving discrete inflaton candidates
    Kill shot:  slow-roll surface is codimension 0 (dense/open in stabilization locus)
    Entry:      GL=12 stabilization → compute ε, η over moduli space → identify intersection with LVS locus
    Depends on: 2A

### 5C. WGC → de Sitter Swampland Coefficient

    Cluster:    H82
    Bet:        WGC-constrained lightest charged state determines instanton tower structure → fixes coefficient c in |∇V| ≥ cV
    Kill shot:  WGC-permitted M_charged gives |∇V|/V < 0.1 at stabilization (too flat)
    Entry:      GL=12 → lightest charged state → instanton sum → gradient of V at stabilization
    Depends on: 2A, 5A

### 5D. Strong-Coupling Flow Equation

    Cluster:    H119
    Bet:        RG-like interpolation tracks V_eff deformation from weak → strong coupling, connecting perturbative type IIB to M-theory regime
    Kill shot:  predictions for moduli masses at g_s ∈ [0.5, 2] deviate >15% from direct interpolation
    Entry:      perturbative + non-perturbative V_eff at multiple g_s values → construct flow → test against M-theory limit
    Depends on: 2A

---

## TIER 6 — Generalized/Extended Geometry

### 6A. M-theory vs Type IIA Yukawa Comparison

    Cluster:    H24
    Bet:        M-theory compactification on GL=12/D₆ generates strictly smaller viable Yukawa set than type IIA, because 11D monodromy imposes additional constraints
    Kill shot:  M-theory texture rank equals or exceeds type IIA rank
    Entry:      compute D₆-invariant Yukawa in M-theory (Gauss-Manin in 11D) vs type IIA → compare ranks
    Depends on: 1A, 1B

### 6B. Generalized Complex Geometry Lift of D₆

    Cluster:    H73
    Bet:        D₆ lifts to generalized complex moduli of CY4 fiber-base decomposition → expanded constraint space → new realistic textures unavailable classically
    Kill shot:  generalized moduli rank = classical rank (no expansion)
    Entry:      B-field deformation of GL=12 → generalized complex structure → D₆ action → constraint counting
    Depends on: 1A

### 6C. Non-Kähler Mirror Symmetry Extension

    Cluster:    H120
    Bet:        mirror symmetry extends to non-Kähler CY via balanced metric → complex structure exchange, preserving worldsheet instantons
    Kill shot:  period vector on proposed mirror dual doesn't correlate with A-model prepotential
    Entry:      construct non-Kähler deformation of GL=12 → attempt mirror construction → period comparison
    Depends on: 4D

---

## TIER 7 — Heterotic/10D/Anomaly

### 7A. Heterotic Orbifold → SM from Fixed-Point Geometry

    Cluster:    H86
    Bet:        orbifold singularity geometry alone (without bundle moduli or Wilson lines) forces su(3)⊕su(2) in ≥50% of studied singularities
    Kill shot:  <50% of singularities exhibit su(3)⊕su(2) from geometry alone
    Entry:      ℤ_n orbifold singularities → E₈ lattice reduction → Kaluza-Klein mode analysis
    Depends on: nothing

### 7B. 10D Tadpole-SUSY Coupling

    Cluster:    H118
    Bet:        10D structure of type IIB is necessitated by coupling between tadpole cancellation and N=2 SUSY in EFT
    Kill shot:  9D compactification yields equal or larger moduli space
    Entry:      dimensional reduction 10D→9D → compare moduli space dimensions and tadpole defects
    Depends on: nothing

### 7C. Anomaly Cancellation as Geometry

    Cluster:    H91
    Bet:        Green-Schwarz c₂(TX₁₀) = c₁(TX₁₀)²/12 partitions CY bases into critical vs non-critical families, and SM-viable fibrations are over-represented in critical family
    Kill shot:  L²-signature of vertical divisor lattice explains <50% of variance in gauge algebra composition
    Entry:      compute c₂ constraint for GL=12 base → check criticality → correlate with SM viability
    Depends on: nothing

---

## TIER 8 — ML Accelerators

### 8A. GNN for Symmetry Prediction

    Cluster:    X1, X2, X4, X5, X15-X19
    Bet:        GNN on reflexive polytope graphs predicts automorphism group type in milliseconds (vs brute-force seconds/minutes)
    Kill shot:  accuracy plateaus <80% on held-out polytopes
    Entry:      10K KS polytopes → automorphisms() → face lattice as graph → GIN/GAT training
    Depends on: nothing (enables 3A at scale)

### 8B. Physics-Informed ML for SM Search

    Cluster:    X4, X7
    Bet:        swampland conditions + moduli constraints + discriminant structure as differentiable regularizers bias ML search toward SM-viable regions
    Kill shot:  regularized models don't outperform unconstrained baselines
    Entry:      encode swampland constraints as loss terms → train on KS database → measure SM-viable discovery rate
    Depends on: 3A (need labeled viability data)

### 8C. RL Agent for CY Selection

    Cluster:    H1, H15, H19, H67, X46
    Bet:        PPO/A3C with entropy regularization discovers h⁰≥3 line bundles ≥3× more efficiently than greedy baselines; meta-learned RL adapts exploration-exploitation to polytope geometry
    Kill shot:  RL discovery rate ≤35% within 300 evaluations
    Entry:      define reward from h⁰(L) → PPO training on low-h¹¹ polytopes → generalization test
    Depends on: nothing

### 8D. Equivariant GNN Architectures

    Cluster:    X17, X18, X39-X41
    Bet:        GNN layers respecting discrete symmetry groups of polytopes improve accuracy and reduce sample complexity by ≥15pp vs non-equivariant
    Kill shot:  equivariant architectures don't outperform standard GNN
    Entry:      construct equivariant graph layers for D₆ → train on polytope classification → compare
    Depends on: 8A

### 8E. CY-BERT: Transformer Pre-Training on Geometric Data

    Cluster:    X44
    Bet:        BERT-like transformer pre-trained on tokenized polytope/Hodge/cohomology "language" learns latent representations encoding discrete symmetry and flavor structure
    Kill shot:  downstream task performance no better than random forest on hand-crafted features
    Entry:      tokenize polytope vertices + Hodge entries + divisor generators → masked pre-training → fine-tune on symmetry prediction
    Depends on: 8A (need training data infrastructure)

### 8F. Interpretable ML for SM Feature Identification

    Cluster:    X48
    Bet:        SHAP/attention attribution on CY3→SM-viability classifiers identifies which geometric features are the actual discriminators (beyond gauge algebra containment)
    Kill shot:  attribution maps highlight only h¹¹, h²¹ (trivial features)
    Entry:      train classifier on 39 fibrations → SHAP on polytope/divisor features → rank by attribution
    Depends on: 2B, 3A (need labeled examples)

### 8G. Few-Shot Meta-Learning for Sparse CY Data

    Cluster:    H65, H68
    Bet:        meta-learning on episodic mini-batches of polytope invariants predicts h¹¹/h⁰≥3 with >80% accuracy from <10 labeled examples per Hodge class
    Kill shot:  <75% F1-score with 5 support examples
    Entry:      prototypical networks on KS polytope features → episodic training → held-out test
    Depends on: nothing

### 8H. Self-Supervised Contrastive Learning on Polytope Graphs

    Cluster:    H66
    Bet:        GraphCL pretraining on unlabeled polytope graphs encodes enough topological signal to achieve ≥90% of fully-supervised performance with 20% of labels
    Kill shot:  transfer-initialized models with 10k labels achieve MAPE >15% on h⁰(L)
    Entry:      augmented divisor-graph pairs → contrastive pretraining → fine-tune with partial labels
    Depends on: 8A

---

## TIER 9 — Complementary Geometric Questions

### 9A. Elliptic Fiber Automorphism → 4D Flavor

    Cluster:    X20, X29
    Bet:        Z_2/Z_3/Z_4/Z_6 fiber automorphisms (from j-invariant special loci) propagate through fibration to constrain 4D flavor symmetries
    Kill shot:  fiber automorphisms trivially lift (no new constraints beyond what base geometry gives)
    Entry:      classify fibrations by j-invariant stratification → catalog fiber automorphisms → trace to 4D Yukawa
    Depends on: 2F

### 9B. Syzygetic Structure → Flavor Group

    Cluster:    X21
    Bet:        syzygy module properties (degrees, Betti numbers, resolution structure) determine flavor symmetry groups via an algorithmic pathway
    Kill shot:  no correlation between syzygy invariants and automorphism group
    Entry:      Macaulay2/Singular computation of syzygy resolutions for GL=12 → extract Betti numbers → compare with D₆
    Depends on: nothing

### 9C. Line Bundle Moduli Symmetry → Flavor Symmetry

    Cluster:    X22
    Bet:        discrete symmetries of line bundle moduli spaces on CY3s generate or constrain the discrete flavor symmetries in 4D, connected via Picard lattice automorphism group
    Kill shot:  Picard lattice symmetries don't descend to physical flavor symmetries
    Entry:      classify line bundle equivalence classes under D₆ action on Picard lattice of GL=12
    Depends on: 1A

### 9D. Flavon Potential from CY Moduli Kähler Potential

    Cluster:    X23
    Bet:        flavon potential terms derive from CY moduli Kähler potentials and flux superpotentials, replacing ad-hoc driving fields with geometric computation
    Kill shot:  derived potential gives wrong vacuum alignment (no phenomenologically viable minimum)
    Entry:      GL=12 Kähler potential → moduli-dependent flavon direction → effective potential → vacuum alignment
    Depends on: 1A, 2A

### 9E. Calabi-Yau Metric Simplification via Discrete Symmetry

    Cluster:    X24
    Bet:        D₆-equivariant constraints on Ricci-flat metric reduce numerical approximation dimensionality significantly
    Kill shot:  symmetry-reduced computation doesn't converge faster than full computation
    Entry:      construct symmetry-reduced metric Ansatz for GL=12 → Donaldson/PhiFlow numerical solution → compare cost
    Depends on: 1A

### 9F. Fermat Hypersurface Symmetries → Period Engineering

    Cluster:    X53
    Bet:        Fermat quintic explicit automorphism group constrains periods; flavor symmetries from Fermat geometry propagate to D₆-invariant families
    Kill shot:  Fermat periods are too special to generalize
    Entry:      full automorphism group of Fermat CY3 → action on H^{2,1} → period structure → map to GL=12
    Depends on: 1B

### 9G. Fixed-Point Loci → Chirality + Yukawa Rank

    Cluster:    X54
    Bet:        fixed-point set topology of discrete automorphisms deterministically constrains chirality spectrum, Yukawa rank, and moduli stabilization requirements
    Kill shot:  fixed-point data doesn't correlate with phenomenological observables
    Entry:      Lefschetz fixed-point theorem applied to D₆ action on GL=12 → equivariant chirality formula
    Depends on: 1A

### 9H. Very Ample vs Birationally Very Ample Divisors

    Cluster:    H108
    Bet:        D₆-invariant divisors failing very ampleness but birationally very ample → base locus contraction → exceptional loci with twisted-sector vanishing → rank reduction in perturbative Yukawa
    Kill shot:  all 5 D₆-invariant Kähler cone generators are very ample
    Entry:      GL=12 Kähler cone → test very ampleness of generators → identify base loci → D₆ fixed points
    Depends on: 1A

### 9I. Complete Intersection vs Non-Complete Intersection Yukawa

    Cluster:    H112
    Bet:        non-CI CY3s with D₆ produce more stringently constrained Yukawa textures than CIs with equivalent automorphism groups
    Kill shot:  NCI D₆-invariant basis count ≥80% of CI count
    Entry:      find matched pair (CI and NCI with D₆) → compare Yukawa constraint counts
    Depends on: 1A

### 9J. Exact Liouville Filling as SM Filter

    Cluster:    H114
    Bet:        exactness of Liouville filling of generic elliptic fiber restricts moduli space of strong fillings → correlates with SM viability at Spearman ρ > 0.3
    Kill shot:  ρ < 0.3 after controlling for Hodge numbers
    Entry:      contact homology computation for GL=12 elliptic fiber → Liouville vector field → exactness score
    Depends on: nothing

### 9K. Holonomy Subgroups → Flavor

    Cluster:    X38
    Bet:        non-generic holonomy (G₂, Spin(7), reduced SU(n) subgroups) on CY3s deterministically constrains available discrete flavor symmetries
    Kill shot:  no correlation between holonomy subgroup structure and flavor group type
    Entry:      catalog discrete subgroups of SU(3) holonomy surviving orbifolding in KS database
    Depends on: nothing

### 9L. Hodge Plot Topology → Symmetry Prediction

    Cluster:    X60
    Bet:        distribution of (h¹¹, h²¹) pairs in KS database predicts which geometries admit discrete symmetry groups capable of generating flavor structure
    Kill shot:  no clustering of high-symmetry polytopes in Hodge plot
    Entry:      overlay symmetry census on Hodge plot → test for clustering
    Depends on: 3A (need symmetry census)

---

## TIER 10 — Pareto/Optimization/Benchmarking

### 10A. NBI Multi-Objective Optimization of GL=12 Moduli

    Cluster:    H109
    Bet:        normal-boundary intersection algorithm on 5D D₆-invariant Kähler space finds Pareto front satisfying LVS viability + Yukawa fit + proton decay suppression simultaneously
    Kill shot:  random sampling with 5000 evaluations outperforms NBI with 500
    Entry:      define 3 objectives on GL=12 moduli space → NBI subproblems → Pareto front extraction
    Depends on: 2A, 1A

### 10B. SM Selection as Portfolio Optimization

    Cluster:    H117
    Bet:        reformulating Level 0-6 filtering as multi-objective optimization exposes redundancy between constraints → Pareto-efficient ranking from topological invariants alone
    Kill shot:  Pareto frontier (top 20%) contains <60% of verified Level 5-6 survivors
    Entry:      collect Level 0-6 pass/fail data for GL=12 family → Pareto ranking → compare with sequential filtering
    Depends on: 3A, 3B, 3E

### 10C. Learning Curve for SM Vacuum Discovery

    Cluster:    H92
    Bet:        cumulative SM-viable discovery rate follows power law with universal exponent α≈0.5-0.7 across heterotic/F-theory campaigns
    Kill shot:  power-law fit R² < 0.70 on in-distribution campaigns
    Entry:      aggregate enumeration campaign data → fit power law → predict undiscovered fraction
    Depends on: 3A (need enumeration data)

### 10D. One-Loop Threshold Corrections as Topological Filter

    Cluster:    H116
    Bet:        one-loop Yukawa corrections from worldsheet instantons and genus-one contributions provide necessary/sufficient topological filter at Level 3
    Kill shot:  >75% of 39 candidates pass Level 1 AND exhibit correct one-loop Yukawa hierarchy
    Entry:      divisor intersection numbers for GL=12 → instanton contributions → Yukawa ratio computation
    Depends on: 2A

---

## TIER 11 — Niche/Long-Shot

### 11A. Lie Point Symmetry of Moduli Geodesics

    Cluster:    H12
    Bet:        continuous Lie point symmetries of D₆-invariant Kähler moduli geodesic equations break to D₆ under α' corrections
    Kill shot:  geodesic Lie algebra doesn't contain D₆ as subgroup
    Entry:      Weil-Petersson metric on 5D subspace → geodesic equations → Lie symmetry analysis → α' correction
    Depends on: 1A

### 11B. Graphene Analogy: D₆ Valley-Spin Representation

    Cluster:    H13
    Bet:        D₆ action on 3 generations decomposes identically to D₆ action on graphene's K/K' valleys + sublattice pseudospin → anomalous Hall transport maps to Yukawa selection rules
    Kill shot:  D₆ decomposition on generations ≠ E₁ ⊕ A₁ (different from graphene)
    Entry:      compare D₆ irrep structure on CY3 generation space with graphene valley-spin analysis
    Depends on: 1A

### 11C. LVS Scalar-Tensor Gravity

    Cluster:    H14
    Bet:        D₆-preserved LVS stabilization constrains α' corrections → Brans-Dicke parameter ω_BD > 40000 → specific Y₂₆₃/Y₃₅₁ Yukawa ratio correlation
    Kill shot:  ω_BD and Yukawa ratio uncorrelated across D₆-invariant moduli space
    Entry:      α' corrections to K = -2ln(V) for GL=12 → ω_BD computation → correlation with Yukawa ratios
    Depends on: 2A, 1A

### 11D. Lepton Flavor Violation from D₆

    Cluster:    H3
    Bet:        D₆ automorphism constrains Wilson coefficients of dimension-6 LFV dipole operator O_μeγ → μ→eγ branching ratio suppressed by O(10⁻¹⁵) relative to unconstrained case
    Kill shot:  26 D₆-invariant Yukawa couplings don't reduce LFV amplitude → same as unconstrained
    Entry:      D₆ Yukawa texture → dimension-6 operator matching → branching ratio computation
    Depends on: 1A, 1B

### 11E. Arc Consistency Filtering on D₆ Moduli

    Cluster:    H27
    Bet:        constraint satisfaction propagation on GL=12 moduli space identifies structurally forbidden Yukawa couplings 3-5× faster than Gröbner basis
    Kill shot:  arc consistency reduces feasible domain by <15%
    Entry:      encode Picard-Fuchs + D₆ constraints as CSP → arc consistency propagation → measure domain reduction
    Depends on: 1B

### 11F. Generalized Matter Curves → Generation Count

    Cluster:    H93
    Bet:        chiral homology h¹(C, O_C) of matter curves in F-theory determines generation count via zero-mode structure
    Kill shot:  h¹(C, O_C) explains <50% of variance in generation number across ≥500 models
    Entry:      extract matter curves from GL=12 F-theory construction → compute h¹(C, O_C) → compare with generation count
    Depends on: 3C

### 11G. Normal Fan → Hodge Diamond Rule

    Cluster:    H98
    Bet:        normal fan combinatorics (codimension-1 face complexity) of reflexive polytopes deterministically constrains h¹¹ of elliptically fibered CY3
    Kill shot:  mean h¹¹ between high/low facet count polytopes differs by <15%
    Entry:      stratify ≥500 d=4 polytopes by facet count → compute h¹¹ for all valid elliptic fibrations → statistical test
    Depends on: nothing

### 11H. Mirror Complex Structure Fixing → Kähler Reduction

    Cluster:    H99
    Bet:        fixing complex structure moduli via flux reduces effective Kähler moduli dimensionality of mirror by factor ∝ number of stabilized moduli
    Kill shot:  reduction <0.3× predicted across ≥5 mirror families
    Entry:      mirror pairs → flux-stabilize complex structure → measure Kähler moduli space dimension of mirror → compare
    Depends on: 1B

---

## Dependency Graph (Critical Path)

```
nothing ──→ 1A ──→ 1B ──→ 1C/1D/1E/1F/1G
                         ↘ 4D/4E/4F/4G/4H
nothing ──→ 2A ──→ 2E/3D/3F/5A-5D/10D
nothing ──→ 2B
nothing ──→ 2C
nothing ──→ 2D
nothing ──→ 2F ──→ 9A
nothing ──→ 2G
nothing ──→ 3A ──→ 3B/8B/8F/9L/10B/10C/4C
nothing ──→ 4A
nothing ──→ 4H (alt path to 1B)
nothing ──→ 4I
```

**Critical path to first publishable result:**
```
1A (D₆ irrep decomposition) → if nontrivial → 1B (monodromy survival) → 1D (GW selection rules)
                                            → 2A (orientifold census)
                                            → 2B (K3 Picard filter)
```

The gate is 1A. If D₆ acts trivially on generations, pivot to Tier 3 (landscape filters) and Tier 8 (ML accelerators) as standalone publication targets.

---

## Carrying Forward

- Raw data: `workspaces/string_physics/outputs/{hypotheses,gaps}/<domain>/`
- Aggregated: `/tmp/athanor_all_candidates.json`
- H{n} = hypotheses ranked by composite_score descending
- X{n} = cross-domain gaps ranked by avg(novelty, tractability, impact) descending, athanor_meta excluded
- Score ceiling at 5.0 (LLM artifact) — ranking is by physics judgment, not raw score
- To add domains: `athanor create-domain` → Stages 1-3 → `athanor cross-domain`
- This document supersedes `TOP_10_RESEARCH_DIRECTIONS.md`
