
for c in 41200 82400 123600 164800 206000
do

python guacamoliency/BENCHMARK_moses.py \
    --input_dir "data/generated/moses_ClearSMILES_corrected_CL_5_checkpoint_$c.csv" \
    --output_dir "reports/data/BENCHMARK/Clearsmiles_moses_CL_100epoch_training.csv"\
    --model_name "MolGPT_moses_ClearSMILES_corrected_characterlevel_5_checkpoint_$c"\
    --number_worker 5
done


