# Carvana Image Segmentation

The task is binary segmentation: predict a mask that separates each vehicle silhouette (foreground) from its background. The study focuses on accuracy, data efficiency, robustness, failure cases, and processing speed.

## Project status

This repository currently contains the project scaffold and locally downloaded dataset archives. Training, evaluation, and experiment implementations are pending; no experimental results are available yet.

| Component | Status |
| --- | --- |
| Dataset and target | Carvana; vehicle silhouette foreground versus background |
| Classical baseline | Required; algorithm to be selected |
| Deep learning model | Required; architecture and framework to be selected |
| Training approach | Training from scratch or fine-tuning; either is permitted |
| Training-data study | Required; nested 25%, 50%, 75%, and 100% subsets |
| Robustness and error analysis | Required; conditions to be defined |
| Ablation study | Optional |
| Literature survey | Approximately 6–8 papers; shortlist pending |
| Environment and compute | Dependencies, hardware, and compute budget to be recorded |
| Final report | CVPR format; six pages of main content plus references |

The supplied assignment identifies its undergraduate section as CS5131, while this project uses CS5031. The project follows the **undergraduate requirements**, not the graduate CS6031 requirements.

## Research objectives

1. Compare one traditional, non-neural segmentation method with one deep learning method.
2. Measure how performance changes as the amount of training data increases.
3. Evaluate robustness under explicitly defined, dataset-relevant perturbations or distribution shifts.
4. Examine actual prediction failures and explain common error patterns.
5. Compare segmentation quality with computational cost.

The classical method must be simpler than a multilayer perceptron. Candidate methods include thresholding with morphology, logistic regression, support vector machines, and random forests. These are options, not finalized choices.

## Repository structure

```text
CS5031-Carvana-Image-Segmentation/
├── README.md
├── .gitignore
├── carvana-image-masking-challenge/  # Local dataset archives; ignored by Git
├── configs/                        # Model, preprocessing, and experiment settings
├── data/
│   ├── raw/                        # Extracted source images and masks; ignored
│   ├── processed/                  # Derived images and masks; ignored
│   └── splits/                     # Versioned train/validation/test and subset manifests
├── src/
│   ├── data/                       # Loading, preprocessing, transforms, and splitting
│   ├── models/
│   │   ├── classical/              # Traditional segmentation baseline
│   │   └── deep/                   # Deep learning segmentation model
│   ├── training/                   # Fitting, validation, and checkpoint selection
│   ├── inference/                  # Loading fitted models and generating predictions
│   ├── evaluation/                 # Metrics, latency, and robustness evaluation
│   ├── analysis/                   # Failure selection and error visualizations
│   └── utils/                      # Shared seed, configuration, and I/O helpers
├── scripts/                        # Thin entry points that orchestrate src/ workflows
├── notebooks/                      # Exploration and inspection
├── tests/                          # Checks for data, masks, metrics, and split integrity
├── outputs/
│   ├── checkpoints/                # Model weights; ignored
│   ├── predictions/                # Predicted masks; ignored
│   ├── metrics/                    # Generated evaluation records; ignored
│   ├── figures/                    # Generated plots and comparisons; ignored
│   └── logs/                       # Run logs and timing records; ignored
└── docs/
    ├── literature/                 # Paper summaries and citation records
    └── reports/
        ├── interim/                # Interim report materials
        └── final/                  # CVPR report sources and selected results
```

Empty directories contain `.gitkeep` placeholders so the structure is retained in Git. The folders describe intended responsibilities; they do not indicate completed implementations.

Keep reusable logic in `src/` and experiment orchestration in `scripts/`. Store experimental parameters in `configs/`. Notebooks should call reusable code when practical so that final experiments can be reproduced without notebook execution order dependencies.

## Dataset handling

Downloaded archives currently reside in `carvana-image-masking-challenge/`. Keep those archives local and extract the required images and masks into `data/raw/` when the data pipeline is implemented. Use `data/processed/` only for derived data, preserving the source files.

- Verify archive contents, image–mask pairing, dimensions, and mask values before preparing splits.
- Create reproducible train, validation, and test partitions from labeled data. Fix the evaluation partitions across all comparisons.
- Check vehicle identity and repeated views before splitting. Keep views of the same vehicle in one partition to prevent identity leakage.
- Save relative sample paths, group identifiers, split membership, and nested-subset membership in `data/splits/`. Record the seed and split-generation procedure.
- Construct the 25%, 50%, 75%, and 100% training subsets from one fixed training partition, keeping smaller subsets inside larger ones. Where samples are grouped by vehicle, preserve those groups and report actual sample and vehicle counts after rounding.
- Apply geometric transforms consistently to each image and its mask. Use nearest-neighbor interpolation for discrete masks and verify that masks remain binary.
- Fit learned preprocessing and baseline parameters using training data only. Use validation data for tuning and checkpoint selection; reserve test data for final evaluation.

Dataset archives, extracted data, and processed data are excluded from Git. Split manifests should contain identifiers and relative paths, not copied images or machine-specific absolute paths. Dataset attribution and access instructions will be documented when the source details are verified.

## Experimental protocol

| Study | Planned comparison | Controls and records |
| --- | --- | --- |
| Standard evaluation | Classical versus deep learning segmentation | Same held-out images, mask definitions, and metric implementation |
| Data efficiency | 25%, 50%, 75%, and 100% training subsets for both methods | Nested membership; fixed validation and test sets; documented tuning and training budgets |
| Robustness | Clean evaluation versus defined challenging conditions | Fixed perturbation settings and seeds; same conditions for both methods; separately labeled results |
| Error analysis | Actual successful and failed predictions | Input, ground truth, prediction, and an error visualization where useful |
| Efficiency | Processing latency for each method | Hardware, resolution, batch size, timing scope, and measurement procedure |

Robustness conditions have not been selected. Illumination changes, noise, blur, or contrast changes are possible candidates, subject to dataset inspection. Any optional ablation should answer a specific question without displacing required experiments.

### Evaluation metrics

The confirmed interim-report metrics are **precision, recall, and latency**, alongside the training-data scaling study.

- **Precision:** `TP / (TP + FP)` — the fraction of predicted foreground pixels that are correct.
- **Recall:** `TP / (TP + FN)` — the fraction of actual foreground pixels recovered.
- **Latency:** milliseconds per image, with hardware, input resolution, batch size, warm-up, timed repetitions, and device synchronization where applicable documented.

Before evaluation, define foreground encoding, prediction threshold selection, aggregation across images, and behavior when a metric denominator is zero. Use the same definitions for both methods. Distinguish model forward-pass time from end-to-end preprocessing, prediction, and postprocessing time.

IoU and Dice/F1 are optional additions that can summarize foreground overlap and the balance between precision and recall. They have not been established as required interim metrics.

## Reproducibility and run records

Each experiment should record:

- Run identifier, source revision, configuration, and random seeds.
- Environment, dependency versions, hardware, and relevant determinism settings.
- Dataset manifests, actual subset sizes, preprocessing, and augmentation settings.
- Model or baseline parameters, optimizer and training settings where applicable, and tuning procedure.
- Validation metrics, checkpoint selection criterion, training duration, and compute budget.
- Test metrics, latency methodology, robustness conditions, and paths to generated artifacts.

Keep large generated artifacts under `outputs/`, which is ignored except for directory placeholders. Copy selected, compact tables and figures into the report directories when they become part of a reviewed project deliverable. Record their originating run identifiers.

Use descriptive names that identify the method, subset, seed, and evaluation condition, for example `classical_train25_seed42_clean`. This is a naming example, not an existing experiment.

### Environment setup

The Python version, training framework, and dependency versions have not yet been selected. An executable setup guide and dependency specification will be added with the implementation. There are currently no training or evaluation commands to run.

## Literature and reporting

Store structured summaries of approximately 6–8 relevant papers in `docs/literature/`. For each paper, record metadata and source links; problem and motivation; architecture and methodology; datasets and experimental setup; reported results and limitations; and relevance to this project. Clearly distinguish results reported by a paper from results produced in this repository.

Use `docs/reports/interim/` for interim materials and `docs/reports/final/` for the final **CVPR-format report with six pages of main content and references outside that limit**. Report actual evidence, including failure cases, compute costs, limitations, and AI-use disclosure.

### AI-use disclosure

AI assistance was used to draft this README and create the initial repository folder scaffold. Update the disclosure as assistance is used for implementation, analysis, or reporting, following the course's disclosure requirements.

## Planned implementation order

1. Inspect dataset contents and pairing, establish vehicle grouping, and define split and subset manifests.
2. Select the classical baseline, deep learning architecture, framework, environment, and compute budget.
3. Implement shared data handling and evaluation, then establish a working classical baseline.
4. Implement deep learning training and checkpoint selection.
5. Run the training-data scaling and robustness studies with consistent evaluation.
6. Analyze real failures and latency, then assemble evidence for the final report.
