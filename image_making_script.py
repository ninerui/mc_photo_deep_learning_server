import os
import time
import json
import logging
import threading
from urllib.request import urlretrieve

import cv2
import requests
import numpy as np
from tensorflow import keras

import conf
from utils import connects, util
from dl_module import image_making_interface, image_quality_assessment_interface

try:
    import absl.logging

    # https://github.com/abseil/abseil-py/issues/99
    logging.root.removeHandler(absl.logging._absl_handler)
    # https://github.com/abseil/abseil-py/issues/102
    absl.logging._warn_preinit_stderr = False
except Exception:
    pass


def image_making_main():
    thread_id = threading.current_thread().getName()
    aesthetic_model = image_quality_assessment_interface.QualityAssessmentModel(
        model_path='./models/weights_mobilenet_aesthetic_0.07.hdf5')
    technical_model = image_quality_assessment_interface.QualityAssessmentModel(
        model_path='./models/weights_mobilenet_technical_0.11.hdf5'
    )
    logging.info("线程ID: {}, 启动完成...".format(thread_id))
    while True:
        params = r_object.rpop_content(conf.res_image_making_name)
        if params:
            params_count = r_object.llen_content(conf.res_image_making_name)
            logging.info("线程ID: {}, 获取到数据, 开始进行打标, 图片打标还剩 {} 条数据!".format(thread_id, params_count))
            params = json.loads(params)
            media_id = params['media_id']
            image_url = params['image_url']
            file_id = params['file_id']
            callback_url = params['callback_url']
            img_path = os.path.join('./', os.path.basename(image_url))
            try:
                start_time = time.time()
                urlretrieve(image_url, img_path)
                download_time = time.time() - start_time

                assessment_start_time = time.time()
                assessment_img = np.asarray(keras.preprocessing.image.load_img(img_path, target_size=(224, 224)))
                aesthetic_value = aesthetic_model.get_res(assessment_img)
                technical_value = technical_model.get_res(assessment_img)
                assessment_time = time.time() - assessment_start_time

                tags = []
                oi_5000_start_time = time.time()
                tags = tags + oi_5000_model.get_tag(img_path)
                oi_5000_time = time.time() - oi_5000_start_time

                raw_img = cv2.imread(img_path)
                ml_1000_start_time = time.time()
                tags = tags + ml_1000_model.get_tag(raw_img)
                ml_1000_time = time.time() - ml_1000_start_time

                ml_11166_start_time = time.time()
                tags = tags + ml_11166_model.get_tag(raw_img)
                ml_11166_time = time.time() - ml_11166_start_time

                logging.info("线程ID: {}, 图片下载地址: {}, 下载时间: {}, 质量时间: {}, oi5000时间: {}, ml1000时间: {}, "
                             "ml11166时间: {}".format(thread_id, image_url, download_time, assessment_time, oi_5000_time,
                                                    ml_1000_time, ml_11166_time))
                data_json = {
                    'mediaId': media_id,
                    'fileId': file_id,
                    'tag': str(tags),
                    'filePath': image_url,
                    'exponent': str({"aesthetic": aesthetic_value, "technical": technical_value}),
                    'identity': str({"isIDCard": 0})
                }
                call_res = requests.post(callback_url, json=data_json)
                logging.info("线程ID: {}, 返回代码: {}, 返回内容: {}".format(thread_id, call_res.status_code, call_res.text))
            except Exception as e:
                logging.exception(e)
            finally:
                os.remove(img_path)
        else:
            time.sleep(1)
        reboot_status = r_object.get_content(local_ip + '_image_making')
        reboot_code = str(reboot_status)
        if reboot_code == '1':
            logging.info('线程ID: {}, 发现服务需要重启, 重启代码: {}'.format(thread_id, reboot_code))
            return


def main():
    # 创建线程
    thread_list = [threading.Thread(target=image_making_main, name=str(i)) for i in range(thread_num)]
    # 启动线程
    for i in thread_list:
        i.start()
    while True:
        thread_status = [i.is_alive() for i in thread_list]
        if sum(thread_status) == 0:
            logging.info('所有线程已停止, 等待重启...')
            return
        time.sleep(20)


if __name__ == '__main__':
    # 创建日志
    util.makedirs("./log")
    log_file = "./log/image_making_script.log"
    util.init_logging(log_file, log_filelevel=logging.INFO, log_streamlevel=logging.INFO, daily=False)
    start_sleep_time = 2
    logging.info("等待 {}s 之后启动程序...".format(start_sleep_time))
    time.sleep(start_sleep_time)

    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
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
    # 获取本地ip
    r_object.set_content(local_ip + '_image_making', "0")

    thread_num = 3
    logging.info("即将开启的线程数: {}".format(thread_num))

    oi_5000_model = image_making_interface.ImageMakingWithOpenImage()

    ml_1000_model = image_making_interface.ImageMakingWithTencent(
        model_path='./models/tencent_1000.pb', label_path='./data/ml_label_1000.txt')

    ml_11166_model = image_making_interface.ImageMakingWithTencent(
        model_path='./models/tencent_11166.pb', label_path='./data/ml_label_11166.txt')
    main()
