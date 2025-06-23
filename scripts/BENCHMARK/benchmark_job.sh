for (( c=16; c<=31; c++ ))
do

python guacamoliency/BENCHMARK_moses.py \
    --input_dir "data/generated/moses_canonical_CL_5_$c.csv" \
    --output_dir 'reports/data/BENCHMARK/BENCHMARK_can_smiles_moses_CL.csv'\
    --model_name "MolGPT_moses_canonical_characterlevel_5"\
    --number_worker 5
done

