# Launch training 

#python guacamoliency/modeling/train.py --tokenizer_path data/tokenizers/moses_canonical/tokenizer.json  --datasets moses_canonical --log_dir reports --dataset_dir data/training_data/moses_canonical.csv --model_save_folder models/trained_moses_canonical --learning_rate 5e-4 --max_steps 10 --batch_size 8 --save_steps 10 --save_total_limit 3


python guacamoliency/modeling/train_scaffolds.py --tokenizer_path data/tokenizersBEP/moses_canonical_mscaffolds  \
    --tokenizer_type "BEP" \
    --datasets moses_canonical_mscaffolds \
    --log_dir reports \
    --dataset_dir data/training_data/moses_canonical.csv \
    --model_save_folder models/trained_moses_canonical_mscaffolds \
    --learning_rate 6e-4 \
    --max_steps 41300 \
    --batch_size 384 \
    --save_steps 10000 \
    --warmup_steps 413 \
    --save_total_limit 3


  


#python run_workflow.py de