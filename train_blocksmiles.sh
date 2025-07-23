python guacamoliency/modeling/train.py \
    --tokenizer_path training_data/tokenizers_blocksmiles \
    --tokenizer_type "simple_blocks" \
    --datasets blocksmiles_simple\
    --log_dir reports \
    --dataset_dir training_data/blocksmiles_simple.csv \
    --model_save_folder models/trained_moses_blocksmiles_simple \
    --learning_rate 6e-4 \
    --max_steps 41300 \
    --batch_size 384 \
    --save_steps 4130\
    --save_total_limit 10\
    --warmup_steps 413
    
