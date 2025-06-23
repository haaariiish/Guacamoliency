# Launch training 

python guacamoliency/modeling/train.py --tokenizer_path data/tokenizers_character_level/moses_canonical_corrected  \
    --tokenizer_type "character_level" \
    --datasets moses_canonical_corrected \
    --log_dir reports \
    --dataset_dir data/training_data/moses_canonical.csv \
    --model_save_folder models/trained_moses_canonical_corrected \
    --learning_rate 6e-4 \
    --max_steps 206500 \
    --batch_size 384 \
    --save_steps 20650\
    --save_total_limit 6 \
    --warmup_steps 2065 #1% du total training set de base

#de base c'est 6e-4 le learning rate

#de base le step est a 41200 et warmup_steps 412
