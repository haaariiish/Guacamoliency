python guacamoliency/modeling/train_scaffolds.py --tokenizer_path data/tokenizersBEP/moses_canonical_corrected  \
    --tokenizer_type "BEP" \
    --datasets moses_canonical_mscaffolds_crashtest \
    --log_dir reports \
    --dataset_dir data/training_data/moses_canonical.csv \
    --model_save_folder models/trained_moses_canonical_mscaffolds_crashtest \
    --learning_rate 6e-4 \
    --max_steps 41300 \
    --batch_size 16 \
    --save_steps 5000 \
    --warmup_steps 413 \
    --save_total_limit 3

