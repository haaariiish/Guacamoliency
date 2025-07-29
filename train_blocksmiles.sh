python guacamoliency/modeling/train.py \
    --tokenizer_path data/tokenizers_blocksmiles/moses_canonical \
    --tokenizer_type "simple_blocks" \
    --datasets blocksmiles_simple\
    --log_dir reports \
    --dataset_dir data/training_data/blocksmiles_simple.csv \
    --model_save_folder models/trained_moses_blocksmiles_simple \
    --learning_rate 6e-4 \
    --max_steps 826000\
    --batch_size 192 \
    --save_steps 8260\
    --save_total_limit 100\
    --warmup_steps 826
    


    
