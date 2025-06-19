for (( c=0; c<30; c++ ))

do
python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES \
    --model_dir 'models/trained_moses_ClearSMILES_character_level/5/final_model' \
    --num_sequence 10000 \
    --temperature 1 \
    --output_dir "data/generated/moses_ClearSMILES_CL_5_$c.csv"

echo "job $c done" 
done

