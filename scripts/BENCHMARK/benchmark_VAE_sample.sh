python guacamoliency/BENCHMARK_moses.py \
    --input_dir data/processed/sample_VAE/onlynew_VAE_latent22.csv \
    --output_dir 'reports/data/BENCHMARK/BENCHMARK_VAE.csv'\
    --model_name "Etienne_VAE_latent_22_onlynew"\
    --number_worker 5

python guacamoliency/BENCHMARK_moses.py \
    --input_dir data/processed/sample_VAE/onlynew_VAE_latent15.csv \
    --output_dir 'reports/data/BENCHMARK/BENCHMARK_VAE.csv'\
    --model_name "Etienne_VAE_latent_15_onlynew"\
    --number_worker 5

python guacamoliency/BENCHMARK_moses.py \
    --input_dir data/processed/sample_VAE/VAE_latent15.csv \
    --output_dir 'reports/data/BENCHMARK/BENCHMARK_VAE.csv'\
    --model_name "Etienne_VAE_latent_15"\
    --number_worker 5

python guacamoliency/BENCHMARK_moses.py \
    --input_dir data/processed/sample_VAE/VAE_latent22.csv \
    --output_dir 'reports/data/BENCHMARK/BENCHMARK_VAE.csv'\
    --model_name "Etienne_VAE_latent_22"\
    --number_worker 5