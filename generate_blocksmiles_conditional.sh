 python guacamoliency/modeling/generate.py \
    --model  blocksmiles \
    --model_dir 'models/trained_moses_blocksmiles_maxweight_blocksmiles/1/final_model' \
    --num_sequence 1000 \
    --temperature 1 \
    --output_dir "data/generated/moses_blocksmiles_example.csv"


python guacamoliency/BENCHMARK_moses.py \
    --input_dir "data/generated/moses_blocksmiles_example.csv" \
    --output_dir 'reports/data/BENCHMARK/BENCHMARK_blocksmiles_example.csv'\
    --model_name "example1"\
    --number_worker 5



for (( c=8260; c<=148680; c+=8260 ))
do
   python guacamoliency/modeling/predict_scaffolds.py \
      --model blocksmiles_conditional\
      --model_dir "models/trained_moses_blocksmiles_conditional_blocksmiles/1/checkpoint-$c" \
      --num_sequence 100 \
      --temperature 1 \
      --output_dir "data/generated/moses_blocksmiles_checkpoint$c.csv"\
      --scaffolds "C1-C=N-C=C-C=1"
done



for (( c=8260; c<=82600; c+=8260 ))
do
   python guacamoliency/modeling/generate.py \
      --model blocksmiles_simple\
      --model_dir "models/trained_moses_blocksmiles_simple_simple_blocks/1/checkpoint-$c" \
      --num_sequence 1000 \
      --temperature 1 \
      --output_dir "data/generated/moses_blocksmiles_checkpoint$c.csv"
done