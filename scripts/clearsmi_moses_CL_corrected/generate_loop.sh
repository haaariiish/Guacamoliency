for c in 41200 82400 123600 164800 206000
do
python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES_corrected \
    --model_dir "models/trained_moses_ClearSMILES_corrected_weightedLoss_character_level/1/checkpoint-$c" \
    --num_sequence 10000 \
    --temperature 1 \
    --output_dir "data/generated/moses_ClearSMILES_corrected_CL_1_temperature1_checkpoint_$c.csv"

echo "job $c done" 
done

python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES_corrected \
    --model_dir "models/trained_moses_ClearSMILES_corrected_character_level/11/final_model" \
    --num_sequence 10000 \
    --temperature 1 \
    --output_dir "data/generated/moses_ClearSMILES_corrected_CL_11.csv"

echo "job $c done"

python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES_corrected \
    --model_dir "models/trained_moses_ClearSMILES_corrected_weightedLoss_character_level/1/final_model" \
    --num_sequence 10000 \
    --temperature 1 \
    --output_dir "data/generated/moses_ClearSMILES_corrected_CL_lossWCE_1_final_model.csv"
