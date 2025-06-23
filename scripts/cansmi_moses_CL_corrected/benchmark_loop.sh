

for (( c=2; c<=7; c++ ))
do
python guacamoliency/BENCHMARK_moses.py \
    --input_dir "data/generated/moses_canoical_corrected_CL_$c.csv" \
    --output_dir 'reports/data/BENCHMARK/Canonical_moses_CL_41200steps_corrected.csv'\
    --model_name "MolGPT_moses_canonical_corrected_characterlevel_$c"\
    --number_worker 5

done

