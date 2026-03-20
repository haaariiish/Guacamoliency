# GuacaMoliency

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Study of saliency map of MolGPT transformer for MOSES and Guacamol dataset

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         guacamoliency and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── guacamoliency   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes guacamoliency a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── generate.py         <- Generate SMILES/SELFIES samples from a trained model
    │   ├── predict_scaffolds.py <- Scaffold-related inference/analysis (see script)
    │   └── train.py            <- Code to train models
    
```

--------

# Environment installation

You can use the environment.yml file to build a conda environment as 

` conda env create -f "environment.yml" -n "guacamoliency_env" `

# Unconditionnal Training script

You can find the `train.py` file in `guacamoliency/modeling/`.

This script fine-tunes a GPT-2 language model (via `transformers.Trainer`) on a CSV dataset.

## Dataset CSV inputs (required columns)

The CSV loaded from `--dataset_dir` must include:
- `SPLIT`: values `train` and `test`
- Token column depending on `--tokenizer_type`:
  - `--tokenizer_type SELFIES`: column `SELFIES`
  - `--tokenizer_type INCHIES`: column `inchkey_encoding`
  - `--tokenizer_type blocksmiles`: columns `SMILES` and `block`
  - otherwise (default tokenization): column `SMILES`

Additional requirement:
- If `--loss_fc` is `Weighted_Cross_Entropy`, column `SMILES` must exist (it is used to compute token weights from `data_set["SMILES"]`).

## CLI arguments (from `modeling/train.py`)

Required:
- `--datasets` (str): dataset name (required by the parser)
- `--dataset_dir` (str): path to the training CSV
- `--tokenizer_path` (str): path to a tokenizer directory for `AutoTokenizer.from_pretrained(...)`
- `--tokenizer_type` (str, required): one of `INCHIES`, `SELFIES`, `blocksmiles` (or anything else for SMILES tokenization)

Optional (defaults shown in code):
- `--log_dir` (str, default: `reports`)
- `--model_save_folder` (str, default: `models/trained_moses_canonical`)
- `--learning_rate` (float, default: `6e-4`)
- `--max_steps` (int, default: `41300`)
- `--batch_size` (int, default: `384`)
- `--save_steps` (int, default: `5000`)
- `--save_total_limit` (int, default: `5`)
- Model config:
  - `--n_embd` (int, default: `256`)
  - `--n_layer` (int, default: `8`)
  - `--n_head` (int, default: `8`)
  - `--resid_pdrop` (float, default: `0.1`)
  - `--embd_pdrop` (float, default: `0.1`)
  - `--attn_pdrop` (float, default: `0.1`)
- Scheduler/training:
  - `--warmup_steps` (int, default: `413`)
  - `--lr_scheduler_type` (str, default: `cosine_with_min_lr`)
  - `--num_workers` (int, default: `10`) for `dataloader_num_workers`
- Loss:
  - `--loss_fc` (str, default: `Cross_Entropy`) or `Weighted_Cross_Entropy`

Note:
- In the current code, the model config uses `args.attn_pdrops` (typo) even though the CLI argument is `--attn_pdrop`. If you hit an AttributeError, this is why.

Example (template):
`python guacamoliency/modeling/train.py --datasets moses_canonical --dataset_dir PATH/TO/data.csv --tokenizer_path PATH/TO/tokenizer_dir --tokenizer_type SELFIES`
# Unconditionnal Sampling script
You can find the `generate.py` file in `guacamoliency/modeling/`.

This script generates sequences from a trained HuggingFace causal LM.

CLI arguments (from `modeling/generate.py`):
- `--model` (str, required by parser): model/dataset name (used only for labeling/help)
- `--model_dir` (str, required): local path to a saved model directory
- `--mscaffolds` (str, default: `False`): enable scaffold decoding (accepts `True`, `true`, `1`)
- `--num_sequence` (int, default: `1000`): number of generated sequences
- `--temperature` (float, default: `1`): sampling temperature
- `--output_dir` (str, required): output CSV path
- `--max_length` (int, default: `15`): currently generation uses `tokenizer.model_max_length` in the script

# Benchmark script
The benchmark script is `guacamoliency/BENCHMARK_moses.py`.
It requires an environment with the `molsets` package installed.

<<<<<<< HEAD
## Benchmark inputs (from `BENCHMARK_moses.py`)

Required:
- `--input_dir` (str): path to a CSV containing a `SMILES` column (non-string entries are filtered out)
- `--output_dir` (str): path to a CSV file where metrics are saved/appended
- `--model_name` (str): name stored in the output row

Optional:
- `--number_worker` (int, default: `1`): `n_jobs` passed to `moses.get_all_metrics`

Important implementation details:
- The script uses `device="cuda"` when calling `moses.get_all_metrics`.
- If `moses.get_all_metrics` raises `ValueError` and the dataset is larger than 10k, the script samples up to 10k SMILES.

## Benchmark output
- `--output_dir` is written as a CSV containing:
  - metric columns returned by `moses.get_all_metrics(...)`
  - `model_name`
  - `sample length` (length after filtering non-string entries)
