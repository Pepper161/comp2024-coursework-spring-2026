# Paper Writing Checklist

## Before Writing the Body

### 1. Freeze the Method Policy

- [ ] Baseline fixed as `RF`
- [ ] Primary methods fixed as `GA`, `PSO`, `SA`
- [ ] Secondary methods fixed as `Tabu Search`, `VNS`
- [ ] Overall comparison table policy fixed
- [ ] Conclusion policy fixed: `best overall` first, `best primary` second

### 2. Complete the Literature Matrix

- [ ] `RF` baseline sources selected
- [ ] `GA` sources selected
- [ ] `PSO` sources selected
- [ ] `SA` sources selected
- [ ] `Tabu Search` positioning sources selected
- [ ] `VNS` positioning sources selected
- [ ] Safe wording written for each method

### 3. Convert Experiment Logs into Paper Tables

- [ ] Table A: overall results for `RF + GA + PSO + SA + Tabu + VNS`
- [ ] Table B: primary comparison for `RF + GA + PSO + SA`
- [ ] Table C: reproducibility and settings summary

Required result columns:

- [ ] `Accuracy`
- [ ] `Precision`
- [ ] `Recall`
- [ ] `F1`
- [ ] `FPR`
- [ ] selected feature count
- [ ] runtime

### 4. Freeze the Figure Plan

- [ ] Figure 1: primary-method `F1 / FPR` comparison
- [ ] Figure 2: feature-count versus detection-quality trade-off
- [ ] Figure 3: runtime comparison
- [ ] Convergence curves moved to appendix unless clearly needed in the main body

### 5. Freeze Core Method Definitions

- [ ] task definition written
- [ ] baseline definition written
- [ ] search representation written
- [ ] fitness function written
- [ ] fairness policy written
- [ ] preprocessing summary written

### 6. Freeze Security Trade-off Rules

- [ ] A method is not called best by `F1` alone
- [ ] `FPR` must be discussed explicitly
- [ ] feature reduction must not come at unacceptable recall loss
- [ ] runtime must be considered in practical interpretation
- [ ] final best-method claims must use balanced judgment

### 7. Write One-Sentence Chapter Goals

- [ ] Introduction one-line goal
- [ ] Related Work one-line goal
- [ ] Problem Formulation one-line goal
- [ ] Algorithms one-line goal
- [ ] Results one-line goal
- [ ] Discussion one-line goal
- [ ] Conclusion one-line goal

## Writing-Phase Checks

- [ ] Do not repeat the `primary` versus `secondary` explanation in every section
- [ ] Show all implemented methods before narrowing interpretation
- [ ] Keep `Tabu Search` and `VNS` visible, not hidden
- [ ] Avoid overclaiming literature support for `Tabu Search` or `VNS`
- [ ] Explain why `FPR` matters operationally
- [ ] Keep the overall-best statement ahead of the primary-best statement

## Final Review Checks

- [ ] Does the paper still match the coursework rubric?
- [ ] Are all method claims supported by the literature matrix?
- [ ] Are overall and primary best-method claims consistent with the results?
- [ ] Are the tables and figures aligned with the text?
- [ ] Is the distinction between empirical strength and literature strength handled honestly?
