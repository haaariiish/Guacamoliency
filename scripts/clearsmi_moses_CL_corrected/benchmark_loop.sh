for (( c=1; c<3; c++ ))
do

python guacamoliency/BENCHMARK_moses.py \
    --input_dir "data/generated/moses_ClearSMILES_corrected_CL_$c.csv" \
    --output_dir 'reports/data/BENCHMARK/Clearsmiles_moses_CL_41200steps_corrected.csv'\
    --model_name "MolGPT_moses_ClearSMILES_corrected_characterlevel_$c"\
    --number_worker 5
done


