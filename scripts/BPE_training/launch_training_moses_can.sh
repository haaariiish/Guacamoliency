# Launch training 

#python guacamoliency/modeling/train.py --tokenizer_path data/tokenizers/moses_canonical/tokenizer.json  --datasets moses_canonical --log_dir reports --dataset_dir data/training_data/moses_canonical.csv --model_save_folder models/trained_moses_canonical --learning_rate 5e-4 --max_steps 10 --batch_size 8 --save_steps 10 --save_total_limit 3


python guacamoliency/modeling/train.py --tokenizer_path data/tokenizersBEP/moses_canonical \
    --tokenizer_type "BEP" \
    --datasets moses_canonical \
    --log_dir reports \
    --dataset_dir data/training_data/moses_canonical.csv \
    --model_save_folder models/trained_moses_canonical \
    --learning_rate 3e-4 \
    --max_steps 82600 \
    --batch_size 384 \
    --save_steps 20000 \
    --warmup_steps 826 \
    --save_total_limit 3\
    --tokenizer_type "BEP"


  


#python run_workflow.py de