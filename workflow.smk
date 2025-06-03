dataset = "moses"
dir_dataset = "data/processed"

rule all:
    input:
        f"{dir_dataset}/{dataset}.csv",
        "notebooks/analysis_tokenizer_vocab_done.flag",
        "notebooks/analysis_tokenizer_vocab_done.ipynb",
        f"models/trained_{dataset}/model.pt"

# Generate a dataset: "moses" or "guacamol" benchmark
rule datasets:
    output:
        f"{dir_dataset}/{dataset}.csv"
    shell:
        f"""conda run -n data_analysis_env \
        python guacamoliency/dataset.py --datasets " + dataset + " --output_dir " + dir_dataset + " > {output}"""

# Generate BPE tokenizer
rule tokenizer:
    output:
        f"data/tokenizers/{dataset}/tokenizer.json"
    shell:
        f"""conda run -n training_env \
        python guacamoliency/generate_tokenizerBEP.py > {output}"""

# Analysis of tokenizer vocab
rule vocab_analysis:
    output:
        "notebooks/analysis_tokenizer_vocab_done.ipynb",
        "notebooks/analysis_tokenizer_vocab_done.flag"
    shell:
        f"""conda run -n data_analysis_env \
        papermill notebooks/analysis_tokenizer_vocab.ipynb {output[0]} && touch {output[1]}"""

# Train a model from scratch using Huggingface
rule train:
    params:
        output_dir = "reports"
    output:
        f"models/trained_{dataset}/model.pt"
    shell:
        f"""
        conda run -n training_env \
        python guacamoliency/modeling/train.py --datasets " + dataset + " --output_dir {params.output_dir} > {output}"""
