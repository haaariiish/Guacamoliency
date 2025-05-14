# Launch training 

#python guacamoliency/modeling/train.py --tokenizer_path data/tokenizers/moses_canonical/tokenizer.json  --datasets moses_canonical --log_dir reports --dataset_dir data/training_data/moses_canonical.csv --model_save_folder models/trained_moses_canonical --learning_rate 5e-4 --max_steps 10 --batch_size 8 --save_steps 10 --save_total_limit 3


python guacamoliency/modeling/train.py --tokenizer_path data/tokenizers/moses_canonical/tokenizer.json  --datasets moses_canonical --log_dir reports --dataset_dir data/training_data/moses_canonical.csv --model_save_folder models/trained_moses_canonical --learning_rate 5e-4 --max_steps 150000 --batch_size 128 --save_steps 10000 --save_total_limit 3


  


#python run_workflow.py de