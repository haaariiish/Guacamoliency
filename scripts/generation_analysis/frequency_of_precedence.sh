python guacamoliency/frequencies_of_tokens.py \
    --model moses_canonical_CL \
    --model_dir 'models/trained_moses_canonical_character_level/1/final_model' \
    --dataset 'data/generated/moses_canonical_CL_1.csv' \
    --output_dir 'reports/data/moses_canonical_CL/1/frequency_precedent.png'\
    
python guacamoliency/frequencies_of_tokens.py \
    --model guacamol_canonical_CL \
    --model_dir 'models/trained_guacamol_canonical_character_level/1/final_model' \
    --dataset 'data/generated/guacamol_canonical_CL_1.csv' \
    --output_dir 'reports/data/guacamol_canonical_CL/1/frequency_precedent.png'\
    