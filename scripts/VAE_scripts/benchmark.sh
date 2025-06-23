
for ((c=0; c<30; c++))
do
python guacamoliency/BENCHMARK_moses.py \
    --input_dir data/processed/sample_VAE/VAE_latent22_$c.csv \
    --output_dir 'reports/data/BENCHMARK/BENCHMARK_VAE_22.csv'\
    --model_name "Etienne_VAE_latent_22"\
    --number_worker 5
done

    
for ((c=0; c<30; c++))
do
python guacamoliency/BENCHMARK_moses.py \
    --input_dir data/processed/sample_VAE/VAE_latent15_$c.csv \
    --output_dir 'reports/data/BENCHMARK/BENCHMARK_VAE_15.csv'\
    --model_name "Etienne_VAE_latent_15"\
    --number_worker 5

done