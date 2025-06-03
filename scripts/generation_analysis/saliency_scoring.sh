python guacamoliency/saliency_scoring.py \
    --model moses_canonical_CL \
    --model_dir 'models/trained_moses_canonical_character_level/1/final_model' \
    --dataset 'data/generated/moses_canonical_CL_1.csv' \
    --output_dir 'reports/data/moses_canonical_CL/1/histo_saliency.png'\
    --threshold 0


python guacamoliency/saliency_scoring.py \
    --model guacamol_canonical_CL \
    --model_dir 'models/trained_guacamol_canonical_character_level/1/final_model' \
    --dataset 'data/generated/guacamol_canonical_CL_1.csv' \
    --output_dir 'reports/data/guacamol_canonical_CL/1/histo_saliency.png'\
    --threshold 0


python guacamoliency/saliency_scoring.py \
    --model moses_canonical_CL \
    --model_dir 'models/trained_moses_canonical_character_level/1/final_model' \
    --dataset 'data/generated/moses_canonical_CL_1.csv' \
    --output_dir 'reports/data/moses_canonical_CL/1/histo_saliency_with_threshold.png'\
    --threshold 1


python guacamoliency/saliency_scoring.py \
    --model guacamol_canonical_CL \
    --model_dir 'models/trained_guacamol_canonical_character_level/1/final_model' \
    --dataset 'data/generated/guacamol_canonical_CL_1.csv' \
    --output_dir 'reports/data/guacamol_canonical_CL/1/histo_saliency_with_threshold.png'\
    --threshold 1

python guacamoliency/saliency_scoring.py \
    --model moses_clearsmi_CL \
    --model_dir 'models/trained_moses_ClearSMILES_character_level/1/final_model' \
    --dataset 'data/generated/moses_ClearSMILES_CL_1.csv' \
    --output_dir 'reports/data/moses_ClearSMILES_CL/1/histo_saliency.png'\
    --threshold 0

python guacamoliency/saliency_scoring.py \
    --model moses_clearsmi_CL \
    --model_dir 'models/trained_moses_ClearSMILES_character_level/1/final_model' \
    --dataset 'data/generated/moses_ClearSMILES_CL_1.csv' \
    --output_dir 'reports/data/moses_ClearSMILES_CL/1/histo_saliency.png'\
    --threshold 1


