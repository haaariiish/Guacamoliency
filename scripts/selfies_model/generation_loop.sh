for i in 0 1 2 3 4 5
do
python guacamoliency/modeling/generate.py \
    --model moses_selfies \
    --model_dir 'models/trained_moses_selfies_SELFIES/1/final_model' \
    --num_sequence 10000 \
    --temperature 1.5 \
    --output_dir "data/generated/moses_SELFIES_1_$i.csv"


done
