python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES \
    --model_dir 'models/trained_moses_ClearSMILES_BEP/2/final_model' \
    --num_sequence 15000 \
    --temperature 1 \
    --output_dir 'data/generated/moses_ClearSMILES_BEP_2.csv'

    python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES \
    --model_dir 'models/trained_moses_ClearSMILES_BEP/1/final_model' \
    --num_sequence 15000 \
    --temperature 1 \
    --output_dir 'data/generated/moses_ClearSMILES_BEP_1.csv'

python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES \
    --model_dir 'models/trained_moses_ClearSMILES_character_level/5/final_model' \
    --num_sequence 10000 \
    --temperature 1 \
    --output_dir 'data/generated/moses_ClearSMILES_CL_1.csv'


python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES \
    --model_dir 'models/trained_moses_ClearSMILES_character_level/1/final_model' \
    --num_sequence 10000 \
    --temperature 1.2 \
    --output_dir 'data/generated/moses_ClearSMILES_CL_1_temp_1dot2.csv'


python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES \
    --model_dir 'models/trained_moses_ClearSMILES_character_level/1/final_model' \
    --num_sequence 10000 \
    --temperature 1.5 \
    --output_dir 'data/generated/moses_ClearSMILES_CL_1_temp_1dot5.csv'

python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES \
    --model_dir 'models/trained_moses_ClearSMILES_BEP/3/final_model' \
    --num_sequence 15000 \
    --temperature 1 \
    --output_dir 'data/generated/moses_ClearSMILES_BEP_3.csv'
