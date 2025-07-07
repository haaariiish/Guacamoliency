python guacamoliency/generate_tokenizerBEP.py --input_dir data/data_tokenizer/moses_ClearSMILES.csv --input moses_ClearSMILES --model_max_length 61

python guacamoliency/generate_tokenizerBEP.py --input_dir data/data_tokenizer/moses_canonical.csv --input moses_canonical --model_max_length 61

python guacamoliency/generate_tokenizerBEP.py --input_dir data/data_tokenizer/guacamol_canonical.csv --input guacamol_canonical --model_max_length 103

#python guacamoliency/generate_tokenizerBEP.py --input_dir data/training_data/guacamol_and_moses_canonical.csv --input guacamol_and_moses_canonical --model_max_length 103 --vocab_size 10000
python guacamoliency/generate_tokenizerBEP.py --input_dir data/training_data/blocksmiles_maxweight.csv --input blocksmiles_maxweight --vocab_size 10000