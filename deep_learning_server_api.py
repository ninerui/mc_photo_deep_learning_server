# 系统包
import json
import logging

# 第三方包
from flask import Flask, request

# 自己的包
import conf
from utils import connects, util

app = Flask(__name__)


@app.route("/face_cluster_api", methods=['POST'])
def face_cluster_api():
    try:
        if request.method == 'POST':
            request_data = str(request.get_data(), encoding='utf-8')
            request_data = eval(request_data)
            user_id = request_data.get('userId')
            if not user_id:
                return json.dumps(conf.status_code['29'])
            base_url = request_data.get('baseUrl')
            if not base_url:
                return json.dumps(conf.status_code['30'])
            callback_url = request_data.get("callbackUrl")
            if not callback_url:
                return json.dumps(conf.status_code['31'])
            handle_result_url = request_data.get("handleResultUrl")
            if not handle_result_url:
                return json.dumps(conf.status_code['32'])
            content = request_data.get('content')
            if not content:
                return json.dumps(conf.status_code['33'])
            oss_running_file = "face_cluster_data/{}/.running".format(user_id)  # 用户状态文件
            exist = oss_bucket.object_exists(oss_running_file)
            if exist:
                return json.dumps(conf.status_code['34'])
            params = {
                'user_id': str(user_id),
                'base_url': base_url,
                'content': content,
                'callback_url': callback_url,
                'handle_result_url': handle_result_url,
            }
            try:
                r_object.lpush_content(conf.res_face_cluster_name, json.dumps(params))
            except Exception as e:
                logging.exception(e)
                return json.dumps(conf.status_code['37'])
            return json.dumps(conf.status_code['00'])
        else:
            return json.dumps(conf.status_code['28'])
    except Exception as e:
        logging.exception(e)
        return json.dumps(conf.status_code['35'])


@app.route("/image_making", methods=['POST'])
def image_making():
    try:
        if request.method == 'POST':
            request_data = str(request.get_data(), encoding='utf-8')
            request_data = eval(request_data)
            request_type = request_data.get('type', None)
            try:
                if request_type:  # 精彩图片合成
                    r_object.lpush_content(conf.res_wonderful_gen_name, json.dumps(request_data))
                else:  # 图片打标
                    r_object.lpush_content(conf.res_image_making_name, json.dumps(request_data))
            except Exception as e:
                logging.exception(e)
                return json.dumps(conf.status_code['37'])
            return json.dumps(conf.status_code['00'])
        else:
            return json.dumps(conf.status_code['28'])
    except Exception as e:
        logging.exception(e)
        return json.dumps(conf.status_code['35'])


if __name__ == '__main__':
    # 创建日志
    util.makedirs("./log")
    util.init_logging(
        "log/deep_learning_server_api.log", log_filelevel=logging.INFO, log_streamlevel=logging.INFO, daily=False)
    local_ip = util.get_local_ip()
    env_code = conf.env_manage.get(local_ip, 0)
    logging.info("获取到的脚本运行环境代码为: {}".format(env_code))
    account_conf = conf.AccountConf(env_code=env_code)
    # 连接oss
    oss_bucket = connects.ConnectALiYunOSS(
        account_conf.accessKeyId, account_conf.accessKeySecret, account_conf.endpoint, account_conf.bucket)
    # 连接redis
    r_object = connects.ConnectRedis(
        account_conf.res_host, account_conf.res_port, account_conf.res_decode_responses, account_conf.res_password)

    # 启动 flask 进程
    if env_code == 1:
        app.run(host='127.0.0.1', port=8081, debug=True)
    else:
        app.run(host='0.0.0.0', port=8081, debug=True)
