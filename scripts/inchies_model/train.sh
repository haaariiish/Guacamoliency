
python guacamoliency/modeling/train.py --tokenizer_path data/tokenizers_inchies/moses_canonical \
    --tokenizer_type "INCHIES" \
    --datasets moses_canonical \
    --log_dir reports \
    --dataset_dir data/training_data/inchies_training_data.csv \
    --model_save_folder models/trained_moses_canonical \
    --learning_rate 6e-4 \
    --max_steps 41300 \
    --batch_size 384 \
    --save_steps 60000 \
    --warmup_steps 413 \
    --save_total_limit 3