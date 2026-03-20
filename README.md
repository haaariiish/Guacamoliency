# GuacaMoliency

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Study of saliency map of MolGPT transformer for MOSES and Guacamol dataset

Supplementary study done for the following [paper](https://www.exemple.com](https://pubs.acs.org/doi/10.1021/acs.jcim.4c02261)

## Project Organization

```
├── LICENSE
├── README.md
├── data
├── env_file
├── guacamoliency
│   ├── __init__.py
│   ├── BENCHMARK_moses.py
│   ├── config.py
│   ├── dataset.py
│   ├── frequencies_of_precedent_tokens.py
│   ├── generate_blocksmiles_tokenizer.py
│   ├── generate_character_based_tokenizer.py
│   ├── generate_selfies_tokenizer.py
│   ├── generate_tokenizerBEP.py
│   ├── modeling
│   │   ├── __init__.py
│   │   ├── generate.py
│   │   ├── predict_scaffolds.py
│   │   ├── train.py
│   │   └── trainer.py
│   ├── plots.py
│   ├── saliency_pair_coherence.py
│   └── saliency_scoring.py
├── models
├── notebooks
├── pyproject.toml
├── references
├── reports
└── scripts
```

--------

# Environment installation

You can use the environment.yml file to build a conda environment as 

` conda env create -f "environment.yml" -n "guacamoliency_env" `

## Pipeline (tokenizer -> train -> génération)

1. Choose and run a tokenizer generator (one of the scripts `guacamoliency/generate_*_tokenizer*.py`). It saves a tokenizer into a reusable folder.
2. Start training with `guacamoliency/modeling/train.py`, pointing `--tokenizer_path` to the folder created in step 1.
3. Generate samples with `guacamoliency/modeling/generate.py`, pointing `--model_dir` to the trained model folder and `--output_dir` to the output CSV.
4. Analyze generated samples (token statistics + saliency):
   - `guacamoliency/frequencies_of_precedent_tokens.py` (inputs: generated CSV via `--dataset`, must contain a `SMILES` column; outputs plots/csv to `--output_dir`).
   - `guacamoliency/saliency_scoring.py` (inputs: `--model_dir` + generated CSV via `--dataset` with `SMILES`; outputs figures/csv to `--output_dir`).
   - `guacamoliency/saliency_pair_coherence.py` (inputs: `--model_dir` + generated CSV via `--dataset` with `SMILES`; optional pair tokens + `--threshold`; outputs csv to `--output_dir`).

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


