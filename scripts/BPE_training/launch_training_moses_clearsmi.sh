# Launch training 

#python guacamoliency/modeling/train.py --tokenizer_path data/tokenizers/moses_ClearSMILES/tokenizer.json  --datasets moses_ClearSMILES --log_dir reports --dataset_dir data/training_data/moses_ClearSMILES.csv --model_save_folder models/trained_moses_ClearSMILES --learning_rate 5e-4 --max_steps 10 --batch_size 8 --save_steps 10 --save_total_limit 3


python guacamoliency/modeling/train.py --tokenizer_path data/tokenizersBEP/moses_ClearSMILES  \
    --tokenizer_type "BEP" \
    --datasets moses_ClearSMILES \
    --log_dir reports \
    --dataset_dir data/training_data/moses_ClearSMILES.csv \
    --model_save_folder models/trained_moses_ClearSMILES \
    --learning_rate 3e-4 \
    --max_steps 82400 \
    --batch_size 384 \
    --save_steps 20000 \
    --save_total_limit 3 \
    --warmup_steps 824\
    --tokenizer_type "BEP"

  


#python run_workflow.py ed

