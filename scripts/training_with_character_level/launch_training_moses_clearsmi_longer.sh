# Launch training 

#python guacamoliency/modeling/train.py --tokenizer_path data/tokenizers/moses_ClearSMILES/tokenizer.json  --datasets moses_ClearSMILES --log_dir reports --dataset_dir data/training_data/moses_ClearSMILES.csv --model_save_folder models/trained_moses_ClearSMILES --learning_rate 5e-4 --max_steps 10 --batch_size 8 --save_steps 10 --save_total_limit 3


python guacamoliency/modeling/train.py --tokenizer_path data/tokenizers_character_level/moses_ClearSMILES  \
    --tokenizer_type "character_level" \
    --datasets moses_ClearSMILES \
    --log_dir reports \
    --dataset_dir data/training_data/moses_ClearSMILES.csv \
    --model_save_folder models/trained_moses_ClearSMILES \
    --learning_rate 6e-4 \
    --max_steps 61200 \
    --batch_size 384 \
    --save_steps 30000 \
    --save_total_limit 3 \
    --warmup_steps 412 #1% du total training steps de base

  
#de base c'est 6e-4 le learning rate

#python run_workflow.py ed

