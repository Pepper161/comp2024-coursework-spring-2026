# Literature Matrix Template

## Purpose

Use this table to lock down what each citation supports before writing `Related Work`, `Discussion`, and method justification paragraphs.

## Column Guide

- `Method`: the method being justified
- `Paper`: citation key or short title
- `Year`: publication year
- `Venue`: journal or conference
- `Dataset`: dataset used in the cited paper
- `Classifier`: main classifier or model
- `Task Type`: `FS`, `HPO`, or `Both`
- `Direct IDS Evidence`: `Yes` or `No`
- `Why It Is Cited`: exact purpose in your paper
- `Safe Wording`: phrasing that stays within the strength of the evidence
- `DOI / Link`: stable source
- `Status`: `Use`, `Backup`, or `Do Not Use`

## Matrix

| Method | Paper | Year | Venue | Dataset | Classifier | Task Type | Direct IDS Evidence | Why It Is Cited | Safe Wording | DOI / Link | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF | Kasongo & Sun | 2020 | Journal of Big Data | UNSW-NB15 | Multiple / RF included | FS / comparison | Yes | Justify RF as a conventional baseline on UNSW-NB15 | RF is a strong conventional baseline commonly used in UNSW-NB15 IDS comparisons. | 10.1186/s40537-020-00379-6 | Use |
| RF | Yin et al. | 2023 | Journal of Big Data | UNSW-NB15 | MLP + RF importance | FS | Yes | Support RF-based feature importance usage in UNSW-NB15 studies | RF is also used as a practical feature-importance mechanism in UNSW-NB15 feature-selection pipelines. | 10.1186/s40537-023-00694-8 | Use |
| RF | Dawood et al. | 2024 | Algorithms | UNSW-NB15 | LR / SVM / DT / RF | classifier comparison | Yes | Support RF as a practically strong conventional IDS model | Several recent UNSW-NB15 studies report RF as one of the strongest conventional classifiers. | 10.3390/a17020064 | Use |
| GA | Halim et al. | 2021 | Computers & Security | IDS datasets incl. UNSW-NB15 | IDS classifiers | FS | Yes | Justify GA as a direct IDS feature-selection benchmark | GA is a widely used benchmark metaheuristic for IDS feature selection. | 10.1016/j.cose.2021.102448 | Use |
| GA | Bakır & Ceviz | 2024 | Arabian Journal for Science and Engineering | IDS datasets | IDS models | HPO / hybrid FS | Yes | Support GA for optimization beyond pure FS | GA is also used in recent IDS optimization studies for hyperparameter tuning and hybrid optimization. | 10.1007/s13369-024-08949-z | Use |
| PSO | Chohra et al. | 2022 | Computers & Security | benchmark IDS datasets incl. UNSW-NB15 | ensemble methods | FS | Yes | Justify PSO as a direct IDS benchmark | PSO is a widely used benchmark metaheuristic for IDS feature selection. | 10.1016/j.cose.2022.102684 | Use |
| PSO | Kilichev & Kim | 2023 | Mathematics | network intrusion detection | 1D-CNN | HPO | Yes | Support GA/PSO as parallel benchmark optimizers | GA and PSO are both established optimization baselines in recent intrusion-detection studies. | 10.3390/math11173724 | Use |
| SA | Huang et al. | 2024 | PeerJ Computer Science | IDS datasets incl. UNSW-NB15 | hybrid optimizer | FS | Yes | Support SA as IDS-relevant, though weaker than GA/PSO | SA has recent IDS-specific support, although the literature is less extensive than for GA and PSO. | 10.7717/peerj-cs.2176 | Use |
| SA | Rosario & Thangadurai | 2016 | International Journal of Computers & Technology | general FS | general | FS | No | Indirect support for SA in feature selection | SA remains a valid feature-selection metaheuristic in general optimization settings. | 10.24297/ijct.v15i2.565 | Backup |
| Tabu | Pacheco et al. | 2023 | Scientific Reports | medical applications | multiple | FS | No | Support Tabu as still active in modern FS research | Tabu Search remains relevant as a memory-based local-search method for feature selection, although direct IDS evidence is limited. | 10.1038/s41598-023-44437-4 | Use |
| Tabu | Huerta et al. | 2002 | Pattern Recognition | general FS | general | FS | No | Classical support for Tabu in FS | Tabu Search has established historical use in feature selection. | 10.1016/S0031-3203(01)00046-2 | Backup |
| VNS | Consoli et al. | 2016 | Information Sciences | high-dimensional FS | general | FS | No | Support VNS as a general high-dimensional FS method | VNS is a valid high-dimensional feature-selection method, but direct recent IDS support is weaker than for GA and PSO. | 10.1016/j.ins.2015.07.041 | Use |
| VNS | Mladenović et al. | 2017 | EURO Journal on Computational Optimization | general optimization | general | optimization theory | No | General method background only | VNS is an established neighborhood-based metaheuristic for complex optimization problems. | 10.1007/s13675-016-0075-x | Backup |

## Writing Rules

- Do not use a paper unless `Why It Is Cited` is specific.
- Do not claim `standard` or `widely used` unless the evidence really supports it.
- Prefer `widely used benchmark metaheuristics` over stronger phrasing such as `the standard benchmark`.
- Mark weak or indirect evidence honestly.

## Priority Order

- Strongest direct support: `RF`, `GA`, `PSO`
- Mid-strength support: `SA`
- Weaker IDS-specific support: `Tabu Search`, `VNS`

## Minimum Use Plan

- `Related Work`: at least 2 papers for `RF`, 2 for `GA/PSO`, 1 for `SA`, 1 careful note each for `Tabu` and `VNS`
- `Discussion`: reuse only the citations that directly support the claims being made
- `Conclusion`: do not introduce new literature claims
