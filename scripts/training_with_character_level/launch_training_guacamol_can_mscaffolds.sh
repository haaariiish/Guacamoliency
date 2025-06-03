# Launch training 

#python guacamoliency/modeling/train.py --tokenizer_path data/tokenizers/guacamol_canonical/tokenizer.json  --datasets guacamol_canonical --log_dir reports --dataset_dir data/training_data/guacamol_canonical.csv --model_save_folder models/trained_guacamol_canonical --learning_rate 5e-4 --max_steps 10 --batch_size 8 --save_steps 10 --save_total_limit 3

python guacamoliency/modeling/train_scaffolds.py --tokenizer_path data/tokenizers_character_level/guacamol_canonical_mscaffolds \
    --tokenizer_type "character_level" \
    --datasets guacamol_canonical_mscaffolds \
    --log_dir reports \
    --dataset_dir data/training_data/guacamol_canonical.csv \
    --model_save_folder models/trained_guacamol_canonical_mscaffolds \
    --learning_rate 6e-4 \
    --max_steps 33200 \
    --batch_size 384 \
    --save_steps 10000 \
    --warmup_steps 332 \
    --save_total_limit 3


  

