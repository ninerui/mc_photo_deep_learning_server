import os
import sys
import time
import shutil
import argparse
import datetime
import subprocess

import conf
from utils import util
from utils import connects


def get_cmd_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cmd", help="use start or restart or stop, 控制后台服务", default='restart')
    parser.add_argument("-s", "--split_log", help="use start or restart or stop, 控制后台服务", default=False)
    return parser.parse_args(argv)


def get_script_status(script_name):
    content = "ps -ef |grep {} |grep -v grep |awk '{{print $2}}'".format(script_name)
    res = subprocess.Popen(content, stdout=subprocess.PIPE, shell=True)
    return res.stdout.readlines()


def start_script(script_name):
    content = "nohup python {} >/dev/null 2>&1 &".format(script_name)
    subprocess.Popen(content, stdout=subprocess.PIPE, shell=True)


def clean_log(path):
    if os.path.exists(path) and os.path.isdir(path):
        file_name_list = []
        for i in range(10):
            date_ = (datetime.date.today() + datetime.timedelta(0 - i)).strftime('%Y-%m-%d')
            file_name_list.append(date_)
        for file in os.listdir(path):
            file_name_sp = file.split('.')
            if len(file_name_sp) > 2:
                file_date = file_name_sp[1]  # 取文件名里面的日期
                if file_date not in file_name_list:
                    abs_path = os.path.join(path, file)
                    os.remove(abs_path)


def main():
    print("当前主机ip: {}, 服务器重启码: {}".format(local_ip, r_object.get_content(local_ip)))
    sub_datetime = datetime.datetime.today()
    hour = sub_datetime.hour
    if hour == 0:
        args.cmd = 'restart'
    cmd = args.cmd
    if cmd == 'daemon':
        python_process = get_script_status('face_cluster_script.py')
        if len(python_process) < 1:
            start_script(script_name)
        return
    elif cmd in ['restart', 'stop']:
        r_object.set_content(local_ip, '1')
        print("设置状态后服务器重启码: {}".format(r_object.get_content(local_ip)))
        while True:
            python_process = get_script_status(script_name)
            if len(python_process) < 1:
                print('程序已经退出...')
                r_object.set_content(local_ip, '0')
                if cmd == 'stop':
                    return
                elif cmd == 'restart':
                    if hour == 0:
                        util.makedirs("./log/face_cluster_log")
                        sub_datetime_str = (sub_datetime + datetime.timedelta(0 - 1)).strftime('%Y-%m-%d')
                        # 分割日志
                        shutil.move(
                            log_file,
                            './log/face_cluster_log/face_cluster_script.{}.log'.format(sub_datetime_str))
                        shutil.move(
                            log_file + ".ERROR",
                            './log/face_cluster_log/face_cluster_script.{}.log.ERROR'.format(sub_datetime_str))
                        clean_log("./log/face_cluster_log")
                    start_script(script_name)
                    return
                return
            else:
                print('程序还在运行...')
                time.sleep(10)


if __name__ == '__main__':
    args = get_cmd_args(sys.argv[1:])
    assert args.cmd in ['restart', 'stop', 'daemon']
    local_ip = util.get_local_ip()
    env_code = conf.env_manage.get(local_ip, 0)
    account_conf = conf.AccountConf(env_code=env_code)
    # 连接redis
    r_object = connects.ConnectRedis(
        account_conf.res_host, account_conf.res_port, account_conf.res_decode_responses, account_conf.res_password)

    script_name = 'face_cluster_script.py'
    log_file = "./log/face_cluster_script.log"
    main()
