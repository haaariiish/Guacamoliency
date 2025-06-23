
python guacamoliency/BENCHMARK_moses.py \
    --input_dir data/generated/moses_canonical_CL_1_temperature_1dot2.csv \
    --output_dir 'reports/data/BENCHMARK.csv'\
    --model_name "MolGPT_moses_canonical_characterlevel_1_temperature_1dot2"\
    --number_worker 5


for i in 0 1 2 3 4 5 
do
python guacamoliency/BENCHMARK_moses.py \
    --input_dir "data/generated/moses_canonical_CL_5_$i.csv" \
    --output_dir 'reports/data/BENCHMARK_can_smiles_moses_CL.csv'\
    --model_name "MolGPT_moses_canonical_characterlevel_5_$i"\
    --number_worker 5
done

python guacamoliency/BENCHMARK_moses.py \
    --input_dir "data/generated/moses_canonical_CL_5.csv" \
    --output_dir 'reports/data/BENCHMARK_can_smiles_moses_CL.csv'\
    --model_name "MolGPT_moses_canonical_characterlevel_5_all"\
    --number_worker 5