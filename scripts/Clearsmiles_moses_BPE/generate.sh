#for (( c=0; c<30; c++ ))
for c in 6 20 
do
python guacamoliency/modeling/generate.py \
    --model moses_ClearSMILES \
    --model_dir 'models/trained_moses_ClearSMILES_BEP/1/final_model' \
    --num_sequence 10000 \
    --temperature 1 \
    --output_dir "data/generated/moses_ClearSMILES_BPE_1_$c.csv"

echo "job $c done" 
done

