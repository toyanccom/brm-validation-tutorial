# Data and code access

## Public materials

Code, the simulated teaching example, expected aggregate outputs, and empirical
aggregate tables are publicly available at
https://github.com/toyanccom/brm-validation-tutorial (version 1.0.0, 5 October 2026).
The simulated teaching example is fully runnable from the supplied code and
pinned direct dependencies. Its generator uses no human records, participant
identifiers, demographic distributions, or parameters fitted to human data.
The generated observation table, predictions, fold audit, paired means, and
comparison figures can all be recreated with `run_tutorial.py`.

The empirical MOT archive supplies executable specifications, source byte
fingerprints, aggregate result tables, and dictionaries. Public summaries are
not the minimum individual data required to refit those models.

## Human data status

The four underlying workbooks are not publicly distributed. The thesis reports
consent before testing and ethics approval, but the available documentation does
not establish permission for unrestricted redistribution of individual records.
This status is an unresolved authorization question, not a finding that a
particular law forbids all sharing. No new consent, waiver, data-use approval,
or anonymization assessment has been obtained by preparing this tutorial.

The requested source files are:

| Source | Relative path for reproduction |
| --- | --- |
| Experiment 1 blur | `Exp1/exp1_blur_merged.xlsx` |
| Experiment 1 contrast | `Exp1/exp1_contrast_merged.xlsx` |
| Experiment 2 blur | `Exp2/exp2_blur_merged.xlsx` |
| Experiment 2 contrast | `Exp2/exp2_contrast_merged.xlsx` |

Hashes in `empirical_archive/results/IJIE/source_verification.csv` identify the
exact inputs. Do not disable verification or treat resaved files as identical.
Public individual-level plot intermediates are also excluded.

## Request route

Send requests to the corresponding author, Mehmet Toyanç Yazgan,
**191215001@stu.gedik.edu.tr**, using the subject
“MOT archived data access request for validation tutorial”. Include:

- Researcher name, institution, and contact address.
- Non-commercial scientific purpose and which results require verification.
- Requested fields and the minimum files or derived data needed.
- Proposed storage, access controls, retention period, and reuse plan.
- Any relevant ethics approval or institutional requirements for your project.

The corresponding author must verify the original consent and institutional
sharing permissions before providing access. A controlled environment or an
authorized minimal de-identified extract may be considered only if allowed by
the actual documentation. No service deadline or automatic approval is promised.
No request is sent by this package.

If access is authorized, the recipient can use the archived reproduction
commands and exact source fingerprints. An authorized derived extract needs a
separate provenance record and may require an explicitly revised loader; it
must not silently bypass the byte check for the original workbooks.

## Before submission

The human authors must settle the permitted empirical access route and update
the availability statement accordingly. The public simulated demonstration
solves reproducibility of the teaching example. It does not solve access to the
underlying empirical observations, establish restricted-data eligibility, or
replace editorial evaluation of the journal's data requirements.
