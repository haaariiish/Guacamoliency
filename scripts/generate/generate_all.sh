for i in 1 2 3 4 5
do
    python guacamoliency/modeling/generate.py \
        --model moses_canonical \
        --model_dir 'models/trained_moses_canonical_character_level/1/final_model' \
        --num_sequence 10000 \
        --temperature 1 \
        --output_dir "data/generated/moses_canonical_CL_1_$i.csv"
done


python guacamoliency/modeling/generate.py \
    --model moses_canonical \
    --model_dir 'models/trained_moses_canonical_BEP/2/final_model' \
    --num_sequence 15000 \
    --temperature 1 \
    --output_dir 'data/generated/moses_canonical_BEP_2.csv'

python guacamoliency/modeling/generate.py \
    --model moses_canonical \
    --model_dir 'models/trained_moses_canonical_BEP/1/final_model' \
    --num_sequence 15000 \
    --temperature 1 \
    --output_dir 'data/generated/moses_canonical_BEP_1.csv'

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


for i in 1 2 3 4 5
do
python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES \
    --model_dir 'models/trained_moses_ClearSMILES_character_level/1/final_model' \
    --num_sequence 10000 \
    --temperature 1 \
    --output_dir "data/generated/moses_ClearSMILES_CL_1_$i.csv"


done