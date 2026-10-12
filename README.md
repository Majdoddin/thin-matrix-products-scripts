# Scripts for "Thin matrix products up to D <= N^0.172 ..."

This repository holds the computer checks of the paper

> Ruhollah Majdoddin, *Thin matrix products up to D <= N^0.172 and lossless reductions for 3SUM, Exact Triangle and
> APSP, with 3SUM over abelian groups and a dichotomy for coordinatewise problems*, 12 October 2026,
> arXiv:XXXX.XXXXX.

The same scripts are the ancillary files (`anc/`) of the arXiv submission; Table 10 of Section 22.1 ("Ancillary
scripts") of the paper lists them. This repository adds, for every script and every mode that the paper cites, the
output of one recorded run (`<script>.out`, or `<script>.<mode>.out` for a script with several modes); three
outputs are still to be added, as the table says. Item numbers below are those of the version of 12 October 2026.

## Kinds

* **P** re-checks a statement that is proved in the text of the paper.
* **C** is part of a computer-assisted proof: Proposition 3.2, the ranks behind Remarks A.6 and A.7, and the
  classifications of Section 20 (Theorems 20.11 and 20.12, Proposition 20.14, Section 20.6).
* **N** computes a numerical value or heuristic evidence that is not proved: the rate values (Remark 6.2), the scores
  and exclusions of Section 11 and Appendix A.1, and the numerical probe after Proposition 3.2.

The paper cites a script by its name, its mode, or the label of a line of its output (for example "output line A");
the labels are the ones the scripts print.

## Requirements and running

Python 3 (tested with Python 3.13), numpy (tested with 2.4), scipy (1.16) and sympy (1.14), and a C compiler `cc`
for `q2_functional4.py`. Run each script from inside this folder:

    python3 FILE [MODE]

Five scripts (`border_identity_scores.py`, `shared_variable_variants.py`, `pure_inner_cases.py`,
`vrana_patterns.py`, `cw_decompositions.py`) import `profile_model.py`, and four (`q2_boolean.py`,
`q2_functional3.py`, `q2_latin.py`, `q2_functional4.py`) import `q2lib.py`; they must lie beside them.

`q2_functional4.py` compiles its two C helpers itself, with `cc -O2 -o enum_functional enum_functional.c` and
`cc -O2 -o locrep locrep.c`, when the executable is missing or older than its source. To build them by hand, run
these two commands in this folder. `q2_functional3.py` writes `q2_functional3.pkl` beside itself, and
`q2_functional4.py local4` writes a temporary file `locrep4_sample.bin`, which it deletes.

Times are wall-clock times of one run on an 8-core desktop computer (Intel i7-4770). `pure_inner_cases.py` uses 7
worker processes, `vrana_patterns.py enum` 8, and `q2_functional3.py` and `q2_functional4.py functional4` 4; the
other scripts are single processes.

## The scripts

| file [mode] | supports (item numbers of the paper) | kind | needs | time | recorded output |
|---|---|---|---|---|---|
| `identity_check.py` | Lemma 3.1, Remark 5.2, the two signs of Coppersmith's identity (Section 3) illegible in the scan | P | sympy | 1 s | `identity_check.out` |
| `squared_check.py` | Theorem 5.1, Lemmas 6.1 and 7.1, every query (Remark 10.2) | P | numpy | 11 s | `squared_check.out` |
| `omv512_check.py` | the index identities of Proposition 10.3 | P | – | 1 s | `omv512_check.out` |
| `example_squared.py` | the worked example of Section 5 (Figure 1) | P | – | 37 s | `example_squared.out` |
| `explicit_constants.py` | Corollary 9.1 (P); Table 4, lines E1, E2 (Table 2), line E3 (Remark 6.2) (N) | P, N | numpy, scipy | 6 s | `explicit_constants.out` |
| `koszul_exact.py` | Remark A.7 (exact rational ranks; controls) | C | – | 1 s | `koszul_exact.out` |
| `koszul_modp.py base` | Remark A.6 | C | numpy | 7 s | `koszul_modp.base.out` |
| `koszul_modp.py excl` | Remark A.7, recomputed modulo a prime | C | numpy | 1 s | `koszul_modp.excl.out` |
| `koszul_modp.py ctrl` | controls of known border rank | C | numpy | 1 s | `koszul_modp.ctrl.out` |
| `border_apolarity.py G222 5` | Proposition 3.2: the border rank of G(2,2,2) is not 5, by weak border apolarity | C | – | 1 s | `border_apolarity.G222_5.out` |
| `border_apolarity.py G221 5`, `G222 6` | controls of Proposition 3.2 | C | – | 1 s, 2 s | `border_apolarity.G221_5.out`, `border_apolarity.G222_6.out` |
| `border_apolarity_family.py S221 5`, `S222 5` | Proposition 3.2, controls (S(2,2,1) has border rank 5: a survivor; S(2,2,2) has border rank at least 6: none) | C | sympy | 6 s, 1 s | `border_apolarity_family.S221_5.out`, `border_apolarity_family.S222_5.out` |
| `border_rank_probe.py` | the numerical probe after Proposition 3.2 (no five-term border decomposition of G(2,2,2) found; an illustration) | N | numpy | 75 s | `border_rank_probe.out` |
| `exact_ceiling.py` | the exact-count limits 0.1784 (squared construction) and 0.1292 (Schönhage), Section 8 | N | numpy, scipy | 12 s | `exact_ceiling.out` |
| `profile_model.py` | the shared scoring model of Section 11; 0.003935 for Schönhage's (3,3) | N | numpy | 2.5 min | `profile_model.out` |
| `schoenhage_scores.py` | Schönhage's family; (3,3) has the largest score | N | – | 15 s | `schoenhage_scores.out` |
| `border_identity_scores.py` | the family's scores 0.005489, 0.004771, 0.004720; the open shapes at most 0.00573 | N | numpy | 25 min | `border_identity_scores.out` |
| `shared_variable_variants.py` | lines H (0.005825), A (0.0049, Remark 5.3), K1 (one product per run: no saving) | N | numpy | 9 min | `shared_variable_variants.out` |
| `pure_inner_cases.py` | the cases R0 ≥ 2 with kn ≤ 30 (Section 11) | N | numpy | 16 min | to be run, output to be added (`pure_inner_cases.out`) |
| `vrana_patterns.py enum` | Vrana's 464 leaf counts, none with a saving (Appendix A.1) | N | numpy | 6 min | to be run, output to be added (`vrana_patterns.enum.out`) |
| `cw_decompositions.py grouped`, `robust`, `wstyle`, `scan 2 4 8` | Coppersmith–Winograd decompositions, no saving (Appendix A.1) | N | numpy | 14 min, 3 min, 42 s, 1 h | `cw_decompositions.grouped.out`, `.robust.out`, `.wstyle.out`; for `scan 2 4 8`: output being recorded, to be added (`cw_decompositions.scan.out`) |
| `thin_exclusion.py [DMAX]` | Corollary A.18: the shapes of the 145 decompositions (Appendix A.9, "Thin full products") | C | – | 1 s | `thin_exclusion.out` |
| `thin_region.py` | the region covered by case (iii) of Proposition A.17 | N | – | 10 min | `thin_region.out` |
| `thin_value.py squares`, `rect`, `sup` | what open thin shapes could be worth; without a mode all three run | N | – | 1 s, 6 s, 1 s | `thin_value.out` (all three modes) |
| `q2lib.py` | library of the four q2 scripts: lattice tests with certificates, gadget search, orbits | C | numpy | – | – |
| `q2_boolean.py` | Theorem 20.11, Table 6: the 256 Boolean relations | C | numpy | 1 s | `q2_boolean.out` |
| `q2_functional3.py` | Theorem 20.12, Table 8: the 19683 functional relations over three letters | C | numpy | 38 s | `q2_functional3.out` |
| `q2_latin.py` | Proposition 20.14(b), (c): the Latin squares of order 4, and L5 | C | numpy | 27 s | `q2_latin.out` |
| `q2_functional4.py functional4` | Proposition 20.14(e), Table 9: the 4^16 functional relations over four letters | C | numpy, cc | 31 min | `q2_functional4.functional4.out` |
| `q2_functional4.py nonfunctional3` | Proposition 20.14(f): all relations over three letters | C | numpy | 19 s | `q2_functional4.nonfunctional3.out` |
| `q2_functional4.py local4` | the counts before Open problem 20.16 (evidence only) | N | numpy, cc | 2.7 min | `q2_functional4.local4.out` |
| `enum_functional.c`, `locrep.c` | C helpers of `q2_functional4.py` (orderly generation of the orbits; relations whose Boolean minors are all representable) | C, N | cc | – | – |

## What to look for in the outputs

    python3 identity_check.py               # checks 1-5 print OK
    python3 squared_check.py                # 0 wrong outputs at 2+2, 3+3, 4+4 levels; the control without the
                                            # split by s fails; every query right
    python3 omv512_check.py                 # both identities hold on 3000 random Boolean instances
    python3 example_squared.py              # the numbers of the example; 0 wrong of 2304 read output strings
    python3 explicit_constants.py           # constants of Corollary 9.1, the rate table, lines E1, E2, E3
    python3 koszul_exact.py                 # S(2,2,2) >= 6, S(2,3,3) >= 8, S(2,3,4) >= 9, S(2,3,5) >= 10,
                                            # S(2,2,3) >= 7; controls S(2,2,1) 5, S(2,3,2) 7
    python3 koszul_modp.py base             # S(1,n,q0') >= n+q0'+1 for n = 2, q0' <= 6 and q0' = 1, n <= 6
    python3 koszul_modp.py excl             # the four bounds of Remark A.7
    python3 koszul_modp.py ctrl             # controls
    python3 border_rank_probe.py            # G(2,2,2) stops at residual 1.0 from every start, like the
                                            # infeasible controls; the feasible controls approach 0
    python3 exact_ceiling.py                # "eps at c_min, gamma=0": 0.17843 (squared), 0.12923 (Schoenhage)
    python3 profile_model.py                # Schoenhage (3,3): eps*gamma = 0.003935
    python3 schoenhage_scores.py            # (3,3) largest, 0.003935
    python3 border_identity_scores.py       # 0.005825 (2,2); 0.005489 (2,3); 0.004771 (3,3); 0.004720 (2,4);
                                            # open shapes 0.005730, 0.004667
    python3 shared_variable_variants.py     # lines H, A, K1
    python3 pure_inner_cases.py             # every case with R0 >= 2 prints "no saving"
    python3 vrana_patterns.py enum          # 464 distinct class profiles, none with a saving
    python3 cw_decompositions.py grouped    # every min Gamma/m is positive
    python3 cw_decompositions.py robust     # every value >= 0
    python3 cw_decompositions.py wstyle     # +0.3812, +0.7278, +1.0744, +1.4210
    python3 cw_decompositions.py scan 2 4 8 # 126 profiles, none negative
    python3 thin_exclusion.py               # (a) not excluded: none; (c) none; exit status 0
    python3 thin_region.py                  # b=2: 23 34 [(34, 2, 35), ...]; b=3: 31 46; b=4: 38 58; ...
    python3 thin_value.py                   # squares: b 2 first k ... 51; rect: b 2 ... o1=42; sup: overall 0.07332
    python3 q2_boolean.py                   # 22 orbits; "relations with REP and t=1 gadget: 0 ; with neither: 0"
    python3 q2_functional3.py               # 84 orbits; non-constant classification: 0; "0 third-class orbits"
    python3 q2_latin.py                     # order 4: 2 orbits, sizes [144, 432]; L5 REP False, gadget False
    python3 q2_functional4.py               # line A: Burnside 167342 = enumerated; line B: 167342 of 167342
                                            # agree; line C: {'REP': 42, 'gadget': 167296, 'neither': 4};
                                            # line E: 0; line I: the 4 third-class orbits, 15552 maps
    python3 q2_functional4.py nonfunctional3  # line A: 22100; line B: 0; line D: 1296 relations in 4 orbits
    python3 q2_functional4.py local4        # line B: 186526704; line C: {'not REP': 10979, 'REP': 9021}

## Data not included

The shapes of the 145 decompositions of full products found by computer search (AlphaTensor, the flip graphs of
Kauers and Moosbauer, AlphaEvolve) are listed in `thin_exclusion.py`; they were read once from the published files
named in Appendix A.1 of the paper, which are not included here. Corollary A.18 needs only the list.

## Licence

MIT; see `LICENSE`.
