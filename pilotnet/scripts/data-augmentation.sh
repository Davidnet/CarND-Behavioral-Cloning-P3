declare -r RAW_DIR="/data/data/"
declare -r AUGMENTED_DIR="/data/data/augmented"

python pilotnet.data_augmentation.py $RAW_DIR $AUGMENTED_DIR True