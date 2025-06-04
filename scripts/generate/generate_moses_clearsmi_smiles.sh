python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES \
    --model_dir 'models/trained_moses_ClearSMILES_BEP/2/final_model' \
    --num_sequence 15000 \
    --temperature 1 \
    --output_dir 'data/generated/moses_ClearSMILES_BEP_2.csv'

python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES \
    --model_dir 'models/trained_moses_ClearSMILES_character_level/1/final_model' \
    --num_sequence 10000 \
    --temperature 1 \
    --output_dir 'data/generated/moses_ClearSMILES_CL_1.csv'


python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES \
    --model_dir 'models/trained_moses_ClearSMILES_character_level/1/final_model' \
    --num_sequence 10000 \
    --temperature 0.8 \
    --output_dir 'data/generated/moses_ClearSMILES_CL_1_temp_0dot8.csv'


python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES \
    --model_dir 'models/trained_moses_ClearSMILES_character_level/1/final_model' \
    --num_sequence 10000 \
    --temperature 1.5 \
    --output_dir 'data/generated/moses_ClearSMILES_CL_1_temp_1dot5.csv'

