# Launch training 

#python guacamoliency/modeling/train.py --tokenizer_path data/tokenizers/moses_canonical/tokenizer.json  --datasets moses_canonical --log_dir reports --dataset_dir data/training_data/moses_canonical.csv --model_save_folder models/trained_moses_canonical --learning_rate 5e-4 --max_steps 10 --batch_size 8 --save_steps 10 --save_total_limit 3


python guacamoliency/modeling/train.py --tokenizer_path data/tokenizers_character_level/moses_canonical \
    --tokenizer_type "character_level_crash_test" \
    --datasets moses_canonical \
    --log_dir reports \
    --dataset_dir data/training_data/moses_canonical.csv \
    --model_save_folder models/trained_moses_canonical \
    --learning_rate 6e-4 \
    --max_steps 41 \
    --batch_size 4 \
    --save_steps 10000 \
    --warmup_steps 413 \
    --save_total_limit 3


  


#python run_workflow.py de