
python guacamoliency/plots.py \
    --our_data "data/generated/moses_canonical_CL_1_temperature_1dot5.csv" \
    --training_data "data/training_data/moses_canonical.csv"\
    --output_dir "reports/figures/moses_canonical_CL/1_temperature_1dot5"\
    --number_worker 5


"""

python guacamoliency/plots.py \
    --our_data "data/generated/both_canonical_BEP_1.csv" \
    --training_data "data/training_data/guacamol_and_moses_canonical.csv"\
    --output_dir "reports/figures/both_canonical_BEP/1"\
    --number_worker 5\
    --sample_training_set 1
"""