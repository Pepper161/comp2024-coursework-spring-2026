# Assessment Rubric Extracted

Source of truth:
- `Assessment Sheet COMP2024 Coursework Spring 2026.pdf`

Extraction note:
- This extraction was produced from `pypdf` text extraction because `pdftoppm` and `pdfplumber` were unavailable in this environment.
- The rubric text is mostly readable, but the PDF contains a few legacy wording fragments such as "portfolio optimization" and "financial theory". Those appear to be template leftovers and are interpreted here as generic optimisation-project expectations, not as literal portfolio-finance requirements for this IDS coursework.

## Coursework Requirements From The Assessment Sheet

The coursework expects the project to:
- preprocess an IDS dataset and explore feature relevance
- apply at least three metaheuristic algorithms for feature selection and/or hyperparameter tuning of one IDS model
- evaluate performance using relevant metrics
- benchmark the optimised methods against at least one non-metaheuristic baseline with default features and default hyperparameters
- analyse trade-offs between detection accuracy, false positives, and number of features
- discuss the performance of the metaheuristics versus the benchmark

The paper is expected to document:
- problem formulation, including objective and constraints
- dataset preprocessing
- details of the selected algorithms
- justification for choosing the algorithms, supported by literature
- performance evaluation with experimental results
- critical evaluation of strengths and weaknesses
- identification of the most effective method based on empirical evidence
- discussion of security trade-offs

Formatting and submission expectations relevant to the paper:
- use the provided conference paper template
- do not change the original formatting and headers
- replace placeholder text with the group’s own content
- convert the final document to PDF before submission
- all final text should be black

## Marking Structure

### Python Code: 30%

#### Functional Correctness and Completeness: 25%

Excellent:
- fully functional and complete with all, or more than, the required algorithms
- algorithms selected are correct and valid for the problem
- produces correct and valid results
- fulfils the required number of algorithms

Satisfactory:
- partially correct implementation of all or some algorithms
- produces mostly correct and valid results with minor issues
- may include fewer than the ideal number of algorithms

Needs Improvement:
- incomplete or incorrect implementation
- major logic errors or incorrect results
- fewer than the minimum required algorithms

#### Code Readability and Documentation: 5%

Excellent:
- well-organized modular code structure
- clear control hierarchy
- well-commented
- standard, systematic naming conventions

Satisfactory:
- proper overall organization with some comments

Needs Improvement:
- poorly structured or inconsistent code
- weak integration
- very limited or no comments

### Documentation / Paper: 50%

#### Introduction and Justification of Chosen Algorithms: 15%

Excellent:
- strong introduction with clear context and background
- strong rationale for chosen algorithms
- depth and strong supporting evidence or references
- well supported by literature review and theoretical background

Satisfactory:
- decent introduction with some context
- justification is present but lacks depth or strong supporting evidence

Needs Improvement:
- lacks key background and context
- claims are unsubstantiated
- weak or missing supporting references

#### Critical Analysis and Discussion of Findings for Performance Evaluation: 25%

Excellent:
- demonstrates deep understanding through thorough discussion of results
- provides meaningful insights
- discussion is well structured and logical
- clearly interprets numerical outcomes and statistical measures
- uses appropriate, well-labeled charts, graphs, or tables
- visuals improve understanding and support findings

Satisfactory:
- covers key points but lacks depth or clarity in places
- interpretation is somewhat generic
- limited or only partially effective visual support

Needs Improvement:
- minimal or no discussion of findings
- weak or missing interpretation of numerical outcomes
- missing, unclear, or unhelpful visuals

#### Insights, Meaningful Conclusions, and Future Work: 10%

Excellent:
- extracts and explains meaningful insights from the results
- connects findings to theory and real-world implications
- provides supported conclusions and plausible future work

Satisfactory:
- some insights are present, but the links to theory or practice are weak or inconsistent

Needs Improvement:
- conclusions are vague, unsupported, or not meaningfully derived from results

### Video Recording: 20%

#### Description of Algorithms, Strengths, and Weaknesses: 20%

Excellent:
- clearly describes underlying concepts and principles
- clearly discusses strengths and weaknesses
- uses findings to justify conclusions
- on time at about 5 to 7 minutes
- answers the Q&A insightfully

Satisfactory:
- some explanation of concepts, but not clearly
- some discussion of pros and cons, but shallow
- timing or Q&A depth is weak

Needs Improvement:
- does not meaningfully discuss algorithms or performance
- poor timing
- no real answers to the required Q&A

## What The Paper Must Explicitly Show To Score Well

To score strongly on the paper component, the draft should explicitly show:
- a clear IDS problem context and why feature selection plus hyperparameter tuning matter
- a justified choice of algorithms grounded in relevant literature, with recent literature prioritized where possible
- a clear experimental protocol with dataset, preprocessing, representation, baseline, optimisation variables, metrics, and fairness controls
- result tables and figures that are clearly labeled, correctly referenced in the text, and actually used in the discussion
- direct interpretation of the reported numbers, not just restatement of tables
- balanced trade-off analysis across detection quality, false positives, feature count, and runtime
- a defensible identification of the best method, with wording aligned to the strength of the evidence
- honest discussion of weaknesses, threats to validity, and remaining uncertainty
- conclusions and future work that follow from the observed results rather than from literature expectations alone

## Practical Reviewer Checklist Derived From The Rubric

When reviewing the paper, check whether it:
- introduces the IDS optimisation problem clearly
- justifies each chosen algorithm with evidence instead of assertion
- defines the optimisation objective and constraints clearly enough to be reproducible
- explains preprocessing and evaluation without leakage ambiguity
- presents the main results and robustness evidence with consistent metrics
- distinguishes raw-metric winners from balanced-trade-off winners where necessary
- uses visuals that materially support the analysis
- discusses strengths, weaknesses, and trade-offs rather than only reporting winners
- avoids claims that exceed the available evidence
- ends with concrete, evidence-based conclusions and realistic future work
