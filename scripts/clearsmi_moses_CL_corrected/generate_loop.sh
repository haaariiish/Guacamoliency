for (( c=1; c<3; c++ ))
do
python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES_corrected \
    --model_dir "models/trained_moses_ClearSMILES_corrected_character_level/$c/final_model" \
    --num_sequence 10000 \
    --temperature 1 \
    --output_dir "data/generated/moses_ClearSMILE_corrected_CL_$c.csv"

echo "job $c done" 
done

