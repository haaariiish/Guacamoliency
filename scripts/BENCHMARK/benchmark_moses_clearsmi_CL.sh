
python guacamoliency/BECNHMARK_moses.py \
    --input_dir data/generated/moses_ClearSMILES_CL_1.csv \
    --output_dir 'reports/data/BENCHMARK.csv'\
    --model_name "MolGPT_moses_ClearSMILES_characterlevel_1"\
    --number_worker 5


python guacamoliency/BECNHMARK_moses.py \
    --input_dir data/generated/moses_ClearSMILES_CL_1_temp_1dot2.csv \
    --output_dir 'reports/data/BENCHMARK.csv'\
    --model_name "MolGPT_moses_ClearSMILES_characterlevel_1_temperature_1dot2"\
    --number_worker 5
