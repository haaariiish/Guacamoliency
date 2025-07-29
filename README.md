# GuacaMoliency

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Study of saliency map of MolGPT transformer for MOSES and Guacamol dataset

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
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
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

# Environment installation

You can use the environment.yml file to build a conda environment as 

` conda env create -f "environment.yml" -n "guacamoliency_env" `

# Unconditionnal Training script

You can find the train.py file in guacamoliency/modeling folders.

Arguments : 
- 
- 
- 
- 
# Unconditionnal Sampling script
You can find the generate.py file in guacamoliency/modeling folders.
Argparser : 
- datasets : indicate only the name of your actual training, needed for the saves of your saves
- dataset_dir : the exact path to your csv file of data 
- log_dir : the path of the folder where you save your logs file
- model_save_folder : the path of your model save folder
- learning_rate : max learning rate for cosine learning rate
- max_steps : how many steps of training
- tokenizer_type : which type of tokenizer, you need to indicate "SELFIES", "blocksmiles" or "SMILES"
- loss_fc : can be the classic cross entropy loss or a custom one. It has to be "Cross_Entropy"  or "Weighted_Cross_Entropy". 

# Benchmark script
The use of the benchmark script has to be done with an other conda environment or virtual environment with the molsets package installed.

Arguments : 
- 
- 
- 
- 