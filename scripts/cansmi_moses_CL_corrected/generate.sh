for c in 41300 82600 123900 165200 206500
do
python guacamoliency/modeling/generate.py \
    --model moses_canonical_corrected \
    --model_dir "models/trained_moses_canonical_corrected_weightedCL_character_level/1/checkpoint-$c" \
    --num_sequence 100 \
    --temperature 1 \
    --output_dir "data/generated/moses_canonical_corrected__weightedCL_1_checkpoint$c.csv"
echo "job $c done" 
done
