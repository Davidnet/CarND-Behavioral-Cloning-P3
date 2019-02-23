declare -r DATA_DIR="/data/data_3"
declare -r JOB_DIR="/models/pilotnet_rel"

python3 -m pilotnet.train --data_dir $DATA_DIR --job_dir $DATA_DIR