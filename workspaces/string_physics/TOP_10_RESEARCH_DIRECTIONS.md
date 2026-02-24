# Top 10 Research Directions — String Physics / SM from F-theory LVS

**Generated:** 2025-01-XX  
**Scope:** 333 candidates (124 hypotheses + 209 cross-domain gaps) across 7 domains  
**Central object:** GL=12/D₆ polytope (h¹¹=17, h²¹=20, KS index 37), D₆ dihedral symmetry  
**Research goal:** Standard Model from F-theory LVS compactifications  

---

## Pattern

Each entry follows this format for forward reference:

```
RANK. SHORT_TITLE
  Source:       domain or cross-domain pair
  Type:         hypothesis | cross-domain gap
  Core idea:    1-2 sentence statement
  Why it ranks: physics justification for ranking
  First step:   single concrete computation you can do next
  CyTools hook: what CyTools call gets you started
  Falsifiable:  what kills it
  Consolidates: which raw candidate IDs this absorbs
```

---

## 1. D₆-Invariant Yukawa Texture → SM Fermion Masses

**Source:** cy_discrete_symmetries (H2, H5, H7, H8, H21, H23, H25)  
**Type:** hypothesis cluster  
**Core idea:** The 26 D₆-invariant Yukawa couplings from GL=12/D₆ constrain the PMNS and CKM matrices. Texture zeros from D₆ irrep decomposition (E₁ ⊕ A₁ on 3 generations) determine neutrino mixing angles, CP phases, and mass-squared differences. This is seven hypotheses that are really one research program.  
**Why it ranks:** This is THE central question. If D₆ symmetry predicts θ₁₂ ≈ 35°, θ₁₃ ≈ 8.5°, and the CKM Cabibbo angle from geometry alone, that's a Nature paper. The 39/39 fibration result says SM gauge factors are generic — the discriminating power lives in the Yukawa texture, not the gauge algebra.  
**First step:** Compute the D₆ representation decomposition on the 3-generation sector. Does D₆ act as E₁ ⊕ A₁ (2+1 split) or 3×A₁ (trivial)? The answer determines whether texture zeros exist at all.  
**CyTools hook:** `Polytope(37).automorphisms()` → verify D₆ generators, then compute their action on (2,1)-form basis via `CalabiYau.intersection_numbers()`.  
**Falsifiable:** If D₆ decomposes as 3×A₁ on generations (trivial action), no texture zeros arise and the entire program collapses.  
**Consolidates:** H2 (θ₁₃), H5 (Majorana CP), H7 (δ_CP vs hierarchy), H8 (θ₁₂), H21 (CKM), H23 (mass hierarchy + θ₁₃), H25 (Δm² differences)

---

## 2. Monodromy-Period Constraint on Flavor Symmetry Survival

**Source:** cy_discrete_symmetries (H4, H6) + cross-domain cy_discrete_symmetries ↔ moduli_periods (X5, X6)  
**Type:** hypothesis + cross-domain  
**Core idea:** The Picard-Fuchs monodromy group acting on GL=12/D₆ periods determines which D₆ generators stabilize the period vector. The dimension of the monodromy-stable subspace controls how many of the 26 Yukawa couplings actually contribute to low-energy physics. This is the mathematical foundation that makes Direction #1 work (or not).  
**Why it ranks:** Without this, Direction #1 is hand-waving. You need to know whether the D₆ symmetry survives moduli stabilization. If monodromy breaks D₆ → Z₂, you get fewer texture zeros than expected. If it preserves full D₆, you get the maximum predictive power.  
**First step:** Compute the Picard-Fuchs operator for the GL=12/D₆ family. Extract monodromy matrices at singular loci. Check which D₆ generators commute with monodromy.  
**CyTools hook:** `CalabiYau.periods()` (if implemented) or extract GKZ system from the polytope lattice points and solve with external PF solver.  
**Falsifiable:** If the monodromy-stable subspace has dimension < 3, or if only trivial D₆ elements stabilize periods, then geometric flavor symmetry doesn't survive to low energy.  
**Consolidates:** H4 (monodromy → discrete flavor), H6 (GUT scale from D₆ + monodromy), X5 (toric CY automorphism → periods), X6 (automorphism → flux constraints)

---

## 3. K3 Picard Lattice as Hypercharge Filter

**Source:** sm_vacuum_selection (H10)  
**Type:** hypothesis  
**Core idea:** The Picard lattice structure of K3 surfaces appearing as bases in elliptically fibered CY3s constrains the U(1)_Y embedding and Yukawa coupling rank. K3 lattice invariants (Picard rank, signature, discriminant) correlate with SM viability beyond mere gauge algebra containment.  
**Why it ranks:** Your 39/39 result says gauge algebra containment is necessary but not sufficient — you need a finer filter. K3 Picard rank is exactly that filter. It's computable today with CyTools, requires no new infrastructure, and directly addresses "why GL=12 is special."  
**First step:** For each of the 39 χ=-6 elliptic fibrations, compute the K3 base Picard lattice. Correlate Picard rank with the number of independent U(1) factors and with Mordell-Weil rank.  
**CyTools hook:** `CalabiYau.divisor_basis()` → extract K3 base divisors → compute intersection form → read off Picard lattice.  
**Falsifiable:** If Picard rank shows no correlation (Spearman ρ < 0.3) with SM viability indicators across the 39 fibrations.  
**Consolidates:** H10

---

## 4. Orientifold Involution Selection for LVS on GL=12

**Source:** moduli_periods (H9)  
**Type:** hypothesis  
**Core idea:** The choice of orientifold involution (O3 vs O7, fixed-point set geometry) on GL=12/D₆ determines whether LVS moduli stabilization is achievable. Different involutions reduce h²¹ differently, altering the flux landscape and the number of stabilizable moduli.  
**Why it ranks:** LVS requires specific volume structure (Swiss cheese or fibered). Not all orientifolds of GL=12 will admit LVS. Classifying which involutions work is a prerequisite for any phenomenological analysis. This is unglamorous but essential.  
**First step:** Enumerate orientifold involutions of GL=12 that are compatible with D₆ symmetry. For each, compute the fixed-point set and the reduced Hodge numbers.  
**CyTools hook:** `CalabiYau.divisors()` → identify involution-invariant divisor combinations → compute h^{1,1}_± and h^{2,1}_±.  
**Falsifiable:** If no D₆-compatible orientifold involution admits an LVS-type volume structure (no large 4-cycle + small blow-up cycle hierarchy).  
**Consolidates:** H9

---

## 5. CY4 → CY3 Reduction Preserving SM Chirality

**Source:** cross-domain sm_vacuum_selection ↔ string_landscape (X12)  
**Type:** cross-domain gap  
**Core idea:** Does the dimensional reduction pathway from CY4 F-theory compactifications through geometric transitions to CY3 compactifications preserve the chirality, generation count, and gauge coupling unification required for SM physics? Can this reduction be systematized?  
**Why it ranks:** F-theory is your starting point. If you can't control what happens when you go from CY4 → CY3, you can't connect F-theory gauge structure to type IIB phenomenology. A systematic map from CY4 Hodge numbers + discriminant loci to CY3 topology would identify which 3-fold vacua inherit SM structure from F-theory.  
**First step:** For GL=12 as an elliptic CY3, identify the CY4 that fibers over it. Compute the CY4 Hodge numbers and discriminant locus. Check whether the F-theory gauge algebra on the CY4 reduces to the known SM factors on the CY3.  
**CyTools hook:** Construct the CY4 as an elliptic fibration over the CY3 base surface. Use `Polytope` methods to build the 5D reflexive polytope.  
**Falsifiable:** If chirality index changes sign or generation count shifts during the transition for the GL=12 family.  
**Consolidates:** X12

---

## 6. DHT Conjecture as SM Vacuum Selector

**Source:** cross-domain moduli_periods ↔ sm_vacuum_selection (X9)  
**Type:** cross-domain gap  
**Core idea:** The Doran-Harder-Thompson conjecture relates CY3 mirror pairs through Tyurin degenerations to quasi-Fano dual geometries. If DHT degenerations impose constraints on moduli spaces accessible to chirality-preserving Yukawa couplings, then Tyurin degeneration structure could be a new SM selection criterion.  
**Why it ranks:** Genuinely novel. Nobody has connected DHT to phenomenology. If Tyurin degeneration loci correspond to Yukawa coupling matrix rank conditions, you'd have a topological reason why some CY3s are better SM candidates. High risk, high reward.  
**First step:** Check whether GL=12/D₆ admits a Tyurin degeneration. If yes, identify the quasi-Fano pieces and compute their periods independently.  
**CyTools hook:** Look for reducible anticanonical divisors in `CalabiYau.effective_cone()` — these signal Tyurin-type degenerations.  
**Falsifiable:** If GL=12 does not admit any Tyurin degeneration, or if CY3s with SM-compatible Yukawa rank are uniformly distributed with respect to DHT structure.  
**Consolidates:** X9

---

## 7. D-Brane Wrapping as D₆ Symmetry-Breaking Mechanism

**Source:** cy_discrete_symmetries (H22) + cy3_discrete_symmetry (H11)  
**Type:** hypothesis cluster  
**Core idea:** D-brane wrappings on non-D₆-invariant divisors break D₆ geometric symmetry by introducing selection rules that forbid specific Yukawa couplings. Discrete torsion H²(D₆, U(1)) ≅ Z₂ on fractional D-branes at orbifold singularities further restricts which of the 26 couplings survive. This is the perturbation theory around the strict D₆ limit of Direction #1.  
**Why it ranks:** Exact D₆ symmetry gives too many texture zeros for realistic masses. You need controlled breaking D₆ → smaller subgroup. D-brane wrapping provides a geometric mechanism for this breaking with computable selection rules. It bridges the idealized symmetry picture with phenomenological reality.  
**First step:** Enumerate tadpole-consistent D-brane stacks on GL=12/D₆ divisors. Classify which divisors are D₆-invariant and which break D₆ to subgroups (Z₂, Z₃, Z₆, trivial).  
**CyTools hook:** `CalabiYau.divisors()` → compute D₆ orbits on divisor classes → `CalabiYau.intersection_numbers()` for tadpole consistency.  
**Falsifiable:** If all tadpole-consistent configurations either preserve full D₆ or break it completely (no intermediate subgroups available).  
**Consolidates:** H22 (D-brane wrapping), H11 (fractional D-branes + discrete torsion)

---

## 8. Discrete Symmetry as Landscape Search Filter

**Source:** cross-domain cy_discrete_symmetries ↔ string_landscape (X7, X8)  
**Type:** cross-domain gap  
**Core idea:** Using CY automorphism groups as a pre-filter before scanning the flux/bundle landscape dramatically reduces the combinatorial explosion. Instead of sampling billions of flux configurations blindly, first restrict to CY3s whose discrete symmetry groups are phenomenologically viable (contain A₄, S₄, D₆, or Δ(27) subgroups), then scan only those.  
**Why it ranks:** Practical strategy with immediate computational payoff. The KS database has ~500M triangulations. Filtering by automorphism group order/type before flux scanning could reduce computation by orders of magnitude. This is a methods paper that enables everything else.  
**First step:** Compute automorphism group orders for all χ=-6 KS polytopes. Correlate symmetry order with h¹¹ and with known three-generation constructions. Establish the pre-filter.  
**CyTools hook:** `Polytope(id).automorphisms()` across all χ=-6 polytopes → build symmetry census.  
**Falsifiable:** If automorphism group structure shows no correlation with SM viability (i.e., SM-viable CY3s are uniformly distributed across symmetry types).  
**Consolidates:** X7 (geometric → string compactification), X8 (moduli space → complex structure)

---

## 9. Fano Sub-Geometry Classification as SM Gate

**Source:** cross-domain sm_vacuum_selection ↔ string_landscape (X14)  
**Type:** cross-domain gap  
**Core idea:** Fano varieties appearing as divisors or fiber components in elliptically fibered CY3s are rigid and finitely classifiable (Mori-Mukai). Systematic enumeration of Fano sub-varieties creates a geometric gate: CY3s whose Fano components match SM-compatible gauge algebra structure pass; others are excluded.  
**Why it ranks:** Fano classification is a solved problem in algebraic geometry. Connecting it to SM phenomenology is new. The rigidity of Fano varieties means this filter is exact (no continuous parameters), unlike most landscape filters. Computationally tractable with existing databases.  
**First step:** For the 39 χ=-6 fibrations, identify all Fano divisors in the resolved toric variety. Match to Mori-Mukai classification. Check whether SM-viable fibrations cluster around specific Fano types.  
**CyTools hook:** `CalabiYau.effective_cone()` → identify Fano divisors (positive anticanonical class) → `Divisor.is_smooth()` + intersection numbers.  
**Falsifiable:** If Fano sub-varieties distribute uniformly across SM-viable and non-viable fibrations (no discriminating power).  
**Consolidates:** X14

---

## 10. GNN Prediction of Discrete Symmetry from Toric Graph

**Source:** cross-domain cy3_discrete_symmetry ↔ cy3_machine_learning (X1, X2, X15-X19)  
**Type:** cross-domain gap cluster  
**Core idea:** Train graph neural networks on reflexive polytope graphs (nodes = lattice points, edges = adjacency) to predict the discrete automorphism group without explicit computation. Equivariant GNN architectures that respect the polytope's own symmetries achieve higher accuracy and require fewer training examples.  
**Why it ranks:** Computing automorphism groups for all 500M KS triangulations is infeasible via brute force. A GNN that predicts symmetry type from the toric graph in milliseconds enables Direction #8 at database scale. This is the computational accelerator for the entire program. Also the most publishable as a standalone ML paper.  
**First step:** Build a training set of ~10K KS polytopes with computed automorphism groups (CyTools can do this). Train a GIN/GAT to predict symmetry group order. Measure accuracy on held-out set.  
**CyTools hook:** `Polytope(id).automorphisms()` on a 10K sample → extract polytope face lattice as graph → train GNN.  
**Falsifiable:** If prediction accuracy plateaus below 80% on held-out polytopes, or if the GNN learns only trivial features (h¹¹ as proxy for symmetry order).  
**Consolidates:** X1 (graph encoder), X2 (pairwise distance), X15-X19 (various GNN approaches)

---

## Execution Order

The directions are ranked by physics value, but the optimal execution order accounts for dependencies:

```
Phase 1 — Foundation (do first, enables everything):
  #4  Orientifold involution classification for LVS on GL=12
  #3  K3 Picard lattice as hypercharge filter
  #8  Discrete symmetry as landscape search filter

Phase 2 — Core Physics (requires Phase 1 results):
  #2  Monodromy-period constraint on flavor symmetry survival
  #1  D₆-invariant Yukawa texture → SM fermion masses
  #7  D-brane wrapping as D₆ symmetry-breaking mechanism

Phase 3 — Extensions (independent, can parallelize):
  #5  CY4 → CY3 reduction preserving SM chirality
  #6  DHT conjecture as SM vacuum selector
  #9  Fano sub-geometry classification as SM gate
  #10 GNN prediction of discrete symmetry from toric graph
```

---

## Carrying Forward

To regenerate or update this list without rebuilding:

1. **Raw data lives in:** `workspaces/string_physics/outputs/`
   - `hypotheses/<domain>/hypothesis_report.json` — within-domain hypotheses
   - `gaps/<domainA>__<domainB>/gap_report.json` — cross-domain bridges

2. **Aggregated snapshot:** `/tmp/athanor_all_candidates.json` (124 hyps + 209 xd gaps)

3. **To add a new domain and re-rank:**
   ```bash
   source .venv-1/bin/activate
   export ATHANOR_WORKSPACE=string_physics
   athanor create-domain --name new_domain --description "..." --seed-queries "..."
   athanor run-stage1 new_domain
   athanor run-stage2 new_domain
   athanor run-stage3 new_domain
   athanor cross-domain --domain new_domain --top 10
   ```
   Then re-aggregate and compare against this list.

4. **Candidate ID scheme:** `H{n}` = hypothesis ranked by composite score, `X{n}` = cross-domain gap ranked by avg(novelty, tractability, impact). These IDs reference the sorted order at generation time, not stable database keys.

5. **Score inflation warning:** All top hypotheses score 5.0 (ceiling effect from LLM scoring). Differentiation comes from physics judgment, not raw scores. The ranking above is based on: (a) direct relevance to GL=12/D₆ SM program, (b) computational tractability with CyTools, (c) falsifiability, (d) novelty vs existing literature.
