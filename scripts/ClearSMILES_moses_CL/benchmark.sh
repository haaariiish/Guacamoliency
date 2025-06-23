for (( c=0; c<=29; c++ ))
do

python guacamoliency/BENCHMARK_moses.py \
    --input_dir "data/generated/moses_ClearSMILES_CL_6_$c.csv" \
    --output_dir 'reports/data/BENCHMARK/BENCHMARK_ClearSMILES_moses_CL_6.csv'\
    --model_name "MolGPT_moses_ClearSMILES_characterlevel_6"\
    --number_worker 5
done


