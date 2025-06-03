
python guacamoliency/plots.py \
    --our_data "data/generated/guacamol_canonical_CL_1.csv" \
    --training_data "data/training_data/guacamol_canonical.csv"\
    --output_dir "reports/figures/guacamol_canonical_CL/1"\
    --number_worker 5

python guacamoliency/plots.py \
    --our_data "data/generated/guacamol_canonical_BEP_1.csv" \
    --training_data "data/training_data/guacamol_canonical.csv"\
    --output_dir "reports/figures/guacamol_canonical_BEP/1"\
    --number_worker 5

python guacamoliency/plots.py \
    --our_data "data/generated/guacamol_canonical_BEP_2.csv" \
    --training_data "data/training_data/guacamol_canonical.csv"\
    --output_dir "reports/figures/guacamol_canonical_BEP/2"\
    --number_worker 5
