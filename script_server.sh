#!/usr/bin/env bash
cd /data/mc_photo_deep_learning_server/
/root/miniconda3/envs/py3/bin/python /data/mc_photo_deep_learning_server/manage_script.py -c $1



# python manage_script.py -c

# crontab
# 3 * * * * /data/mc_photo_deep_learning_server/script_server.sh daemon

