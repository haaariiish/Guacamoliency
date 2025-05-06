

# generate a dataset, "moses" or "guacamol" Benchmark

rule datasets:
    params:
        datasets : "moses" 
    , output_dir : "data/interim"
    shell:
        "python guacamoliency/dataset.py --datasets {params.datasets} --output_dir {params.output_dir}"

# generate BPE tokenizer
rule tokenizer:
    shell:
        "python guacamoliency/generate_tokenizerBEP.py"

# analysis of tokenizer vocab 

rule vocab_analysis:
    notebook:
        "notebooks/analysis_vocab.ipynb"



# Then train a model from scratch, using huggingface

# training script
rule train:
    params:
        datasets : "moses",
        output_dir : "reports"
        
    shell:
        "python guacamoliency/modeling/train.py --datasets {params.datasets} --output_dir {params.output_dir}"

# Restera ensuite la generation de SMILES et les analyses de validité etc