python guacamoliency/modeling/predict_scaffolds.py \
    --model guacamol_canonical_BEP_mscaffolds \
    --model_dir 'models/trained_guacamol_canonical_mscaffolds_BEP/1/final_model' \
    --num_sequence 100\
    --temperature 1 \
    --output_dir 'data/generated/guacamol_canonical_BEP_mscaffolds_completion.csv'\
    --scaffolds "c1cnc(N2CCN(CCCOc3ccc(-c4nc5ccccc5o4)cc3)CC2)nc1" \
    --max_length 206
    

python guacamoliency/modeling/predict_scaffolds.py \
    --model guacamol_canonical_BEP_mscaffolds \
    --model_dir 'models/trained_guacamol_canonical_mscaffolds_BEP/1/final_model' \
    --num_sequence 100\
    --temperature 1 \
    --output_dir 'data/generated/guacamol_canonical_BEP_mscaffolds_completion2.csv'\
    --scaffolds "O=C1C=C2C3CCCCC3CCC2C2CCC3CCCCC3C12" \
    --max_length 206
    
