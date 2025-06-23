for c in 41200 82400 123600 164800 206000
do
python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES_corrected \
    --model_dir "models/trained_moses_ClearSMILES_corrected_character_level/5/checkpoint-$c" \
    --num_sequence 10000 \
    --temperature 1 \
    --output_dir "data/generated/moses_ClearSMILES_corrected_CL_5_checkpoint_$c.csv"

echo "job $c done" 
done

