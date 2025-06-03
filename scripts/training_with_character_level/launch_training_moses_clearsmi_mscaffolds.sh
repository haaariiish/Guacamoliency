# Launch training 

#python guacamoliency/modeling/train.py --tokenizer_path data/tokenizers/moses_ClearSMILES/tokenizer.json  --datasets moses_ClearSMILES --log_dir reports --dataset_dir data/training_data/moses_ClearSMILES.csv --model_save_folder models/trained_moses_ClearSMILES --learning_rate 5e-4 --max_steps 10 --batch_size 8 --save_steps 10 --save_total_limit 3


python guacamoliency/modeling/train_scaffolds.py --tokenizer_path data/tokenizers_character_level/moses_ClearSMILES_mscaffolds \
    --tokenizer_type "character_level" \
    --datasets moses_ClearSMILES_mscaffolds \
    --log_dir reports \
    --dataset_dir data/training_data/moses_ClearSMILES.csv \
    --model_save_folder models/trained_moses_ClearSMILES_mscaffolds \
    --learning_rate 6e-4 \
    --max_steps 41200 \
    --batch_size 384 \
    --save_steps 10000 \
    --save_total_limit 3 \
    --warmup_steps 412

  


#python run_workflow.py ed

