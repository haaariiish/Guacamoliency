for (( c=7; c<13; c++ ))
do
python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES \
    --model_dir "models/trained_moses_ClearSMILES_character_level/$c/final_model" \
    --num_sequence 10000 \
    --temperature 1 \
    --output_dir "data/generated/moses_ClearSMILES_CL_$c.csv"

echo "job $c done" 
done

