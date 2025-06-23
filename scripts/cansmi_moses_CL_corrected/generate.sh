for (( c=1; c<=7; c++ ))
do
python guacamoliency/modeling/generate.py \
    --model moses_canonical_corrected \
    --model_dir "models/trained_moses_canonical_corrected_character_level/$c/final_model" \
    --num_sequence 10000 \
    --temperature 1 \
    --output_dir "data/generated/moses_canonical_corrected_CL_$c.csv"
echo "job $c done" 
done
