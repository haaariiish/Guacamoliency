
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

python guacamoliency/BECNHMARK_moses.py \
    --input_dir data/processed/moses_ClearSMILES_CL_1_novelty100.csv \
    --output_dir 'reports/data/BENCHMARK.csv'\
    --model_name "MolGPT_moses_ClearSMILES_characterlevel_1_without_old"\
    --number_worker 5


for i in 1 2 3 4 5
do
python guacamoliency/BECNHMARK_moses.py \
    --input_dir "data/generated/moses_ClearSMILES_CL_1_$i.csv" \
    --output_dir 'reports/data/BENCHMARK.csv'\
    --model_name "MolGPT_moses_ClearSMILES_characterlevel_1_$i"\
    --number_worker 5
done