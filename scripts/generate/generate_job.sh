
for (( c=16; c<=31; c++ ))
do
python guacamoliency/modeling/generate.py \
    --model moses_canonical \
    --model_dir 'models/trained_moses_canonical_character_level/5/final_model' \
    --num_sequence 10000 \
    --temperature 1 \
    --output_dir "data/generated/moses_canonical_CL_5_$c.csv"

echo "job $c done" 
done

