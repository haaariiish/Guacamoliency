python guacamoliency/generate_character_based_tokenizer.py --input_dir data/data_tokenizer/moses_ClearSMILES.csv --input moses_ClearSMILES_corrected

python guacamoliency/generate_character_based_tokenizer.py --input_dir data/data_tokenizer/guacamol_canonical.csv --input guacamol_canonical_corrected

python guacamoliency/generate_character_based_tokenizer.py --input_dir data/data_tokenizer/moses_canonical.csv --input moses_canonical_corrected 

#python guacamoliency/generate_character_based_tokenizer.py --input_dir data/training_data/guacamol_and_moses_canonical.csv --input guacamol_and_moses_canonical --model_max_length 103
python guacamoliency/generate_character_based_tokenizer.py --input_dir data/training_data/blocksmiles_maxweight.csv --input moses_blocksmiles_maxweight