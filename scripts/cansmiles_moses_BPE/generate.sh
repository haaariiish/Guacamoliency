for (( c=0; c<30; c++ ))
do
python guacamoliency/modeling/generate.py \
    --model moses_canonical \
    --model_dir 'models/trained_moses_canonical_BEP/1/final_model' \
    --num_sequence 10000 \
    --temperature 1 \
    --output_dir "data/generated/moses_canonical_BPE_1_$c.csv"

echo "job $c done" 
done

