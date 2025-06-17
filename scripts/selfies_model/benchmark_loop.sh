for ((c=0; c<6; c++))
do

python guacamoliency/BENCHMARK_moses.py \
    --input_dir "data/generated/moses_SELFIES_1_$c.csv" \
    --output_dir 'reports/data/BENCHMARK/BENCHMARK_SELFIES.csv'\
    --model_name "gpt_SELFIES_temperature_1.5"\
    --number_worker 5

done
