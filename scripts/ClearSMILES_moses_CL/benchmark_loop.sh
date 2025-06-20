for (( c=7; c<13; c++ ))
do

python guacamoliency/BENCHMARK_moses.py \
    --input_dir "data/generated/moses_ClearSMILES_CL_$c.csv" \
    --output_dir 'reports/data/BENCHMARK/Clearsmiles_moses_CL_sametraining.csv'\
    --model_name "MolGPT_moses_ClearSMILES_characterlevel_$c"\
    --number_worker 5
done


