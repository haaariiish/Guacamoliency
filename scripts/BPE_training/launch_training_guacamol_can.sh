# Launch training 

#python guacamoliency/modeling/train.py --tokenizer_path data/tokenizers/guacamol_canonical/tokenizer.json  --datasets guacamol_canonical --log_dir reports --dataset_dir data/training_data/guacamol_canonical.csv --model_save_folder models/trained_guacamol_canonical --learning_rate 5e-4 --max_steps 10 --batch_size 8 --save_steps 10 --save_total_limit 3

python guacamoliency/modeling/train.py --tokenizer_path data/tokenizersBEP/guacamol_canonical \
    --tokenizer_type "BEP" \
    --datasets guacamol_canonical \
    --log_dir reports \
    --dataset_dir data/training_data/guacamol_canonical.csv \
    --model_save_folder models/trained_guacamol_canonical \
    --learning_rate 3e-4 \
    --max_steps 66400 \
    --batch_size 384 \
    --save_steps 20000 \
    --warmup_steps 664 \
    --save_total_limit 3 \
    --tokenizer_type "BEP"


  

