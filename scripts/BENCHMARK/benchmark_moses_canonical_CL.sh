
python guacamoliency/BECNHMARK_moses.py \
    --input_dir data/generated/moses_canonical_CL_1_temperature_1dot2.csv \
    --output_dir 'reports/data/BENCHMARK.csv'\
    --model_name "MolGPT_moses_canonical_characterlevel_1_temperature_1dot2"\
    --number_worker 5


for i in 2 3 
do
python guacamoliency/BECNHMARK_moses.py \
    --input_dir "data/generated/moses_canonical_CL_$i.csv" \
    --output_dir 'reports/data/BENCHMARK.csv'\
    --model_name "MolGPT_moses_canonical_characterlevel_$i"\
    --number_worker 5
done