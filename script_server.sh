#!/usr/bin/env bash
conda activate py3
cd /data/mc_photo_deep_learning_server/
python manage_script.py -c $1





# crontab
# 3 * * * * bash -i /data/mc_photo_deep_learning_server/script_server.sh daemon

