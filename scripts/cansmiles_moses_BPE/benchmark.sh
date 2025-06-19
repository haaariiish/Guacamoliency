
for (( c=0; c<=29; c++ ))
do

python guacamoliency/BENCHMARK_moses.py \
    --input_dir "data/generated/moses_canonical_BPE_1_$c.csv" \
    --output_dir 'reports/data/BENCHMARK/BENCHMARK_canonical_moses_BPE.csv'\
    --model_name "MolGPT_moses_canonical_BPE_1"\
    --number_worker 5
done

