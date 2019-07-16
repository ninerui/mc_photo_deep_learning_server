import os
import time
import pickle
import logging
import threading
from urllib.request import urlretrieve

import cv2
import numpy as np
import requests
from tensorflow import keras

import conf
from utils import connects, util, image_tools
from dl_module import face_cluster_interface
from dl_module import face_emotion_interface
from dl_module import image_quality_assessment_interface
from dl_module import image_making_interface, face_detection_interface, face_recognition_interface


class FilePathConf:
    def __init__(self, user_id):
        self.oss_running_file = "face_cluster_data/{}/.running".format(user_id)
        self.oss_suc_img_list_file = "face_cluster_data/{}/suc_img_list.pkl".format(user_id)
        oss_bucket.get_set_object(self.oss_suc_img_list_file, [])
        self.oss_face_id_with_label_file = "face_cluster_data/{}/face_id_with_label.pkl".format(user_id)
        oss_bucket.get_set_object(self.oss_face_id_with_label_file, {})
        self.oss_face_data_file = "face_cluster_data/{}/face_data.pkl".format(user_id)
        oss_bucket.get_set_object(self.oss_face_data_file, [])

        # self.user_path = os.path.join(conf.output_dir, str(user_id))
        # util.makedirs(self.user_path)
        # subdir = datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d-%H%M%S')
        # self.img_save_path = os.path.join(self.user_path, 'save_images_{}'.format(subdir))
        # util.makedirs(self.img_save_path)
        # self.face_image_path = os.path.join(self.user_path, "face_images_{}".format(subdir))
        # util.makedirs(self.face_image_path)


def image_resize(image):
    m = min(image.shape[0], image.shape[1])
    f = 320.0 / m
    if f < 1.0:
        image = cv2.resize(image, (int(image.shape[1] * f), int(image.shape[0] * f)))
    return image, min(f, 1.)


def call_url_func(user_id, callback_url, data_json):
    call_count = 0
    call_suc_status = False
    while call_count < 20:
        try:
            call_res = requests.post(callback_url, json=data_json)
            if int(call_res.status_code) == 200:
                call_suc_status = True
                logging.info("用户ID: {}, call_status: {}".format(user_id, call_res.text))
                return call_suc_status
            else:
                logging.error("用户ID: {}, call_status: {}".format(user_id, call_res.text))
                time.sleep(9)
                call_count += 1
        except Exception as e:
            logging.exception(e)
            call_count += 1
            time.sleep(9)
            continue
    return call_suc_status


class FaceClusterThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, thread_name):
        threading.Thread.__init__(self)
        self.thread_name = thread_name

    def pr_log(self, content, style='info'):
        log_content = "线程名: {}, {}".format(self.thread_name, content)
        if style == 'info':
            logging.info(log_content)
        elif style == "error":
            logging.error(log_content)
        elif style == 'exception':
            logging.exception(content)

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        while True:
            face_user_key = r_object.rpop_content(conf.redis_face_info_key_list)
            user_id = face_user_key.split('-')[1]
            oss_running_file = "face_cluster_data/{}/.running".format(user_id)
            exist = oss_bucket.object_exists(oss_running_file)
            if exist:
                time.sleep(2)
                continue
            oss_bucket.put_object(oss_running_file, 'running')
            oss_suc_img_list_file = "face_cluster_data/{}/suc_img_list.pkl".format(user_id)
            oss_bucket.get_set_object(oss_suc_img_list_file, set())
            oss_face_id_with_label_file = "face_cluster_data/{}/face_id_with_label.pkl".format(user_id)
            oss_bucket.get_set_object(oss_face_id_with_label_file, dict())
            oss_face_data_file = "face_cluster_data/{}/face_data.pkl".format(user_id)
            oss_bucket.get_set_object(oss_face_data_file, list())

            suc_parser_img_set = pickle.loads(oss_bucket.get_object(oss_suc_img_list_file).read())
            face_id_label_dict = pickle.loads(oss_bucket.get_object(oss_face_id_with_label_file).read())
            old_data = pickle.loads(oss_bucket.get_object(oss_face_data_file).read())

            face_data = []
            success_image_set = set()
            while True:
                data_ = r_object.rpop_content(face_user_key)
                if not data_:
                    break
                data_ = pickle.loads(data_)
                media_id = data_.get('face_id', "").split('_')[0]
                face_data.append(pickle.loads(data_))
                success_image_set.add(media_id)
            if len(face_data) == 0:
                continue
            face_data = old_data + face_data
            start_cluster_time = time.time()
            call_res_dict, face_id_label_dict = face_cluster_interface.cluster_face_func(
                face_data, user_id, face_id_label_dict)
            logging.info('用户ID: {}, 聚类人脸耗时: {}'.format(user_id, time.time() - start_cluster_time))

            oss_bucket.put_object(oss_face_id_with_label_file, pickle.dumps(face_id_label_dict))
            oss_bucket.put_object(oss_face_data_file, pickle.dumps(face_data))
            oss_bucket.put_object(oss_suc_img_list_file, pickle.dumps(success_image_set | suc_parser_img_set))
            exist = oss_bucket.object_exists(oss_running_file)
            if exist:
                oss_bucket.delete_object(oss_running_file)
            # 开始回调结果数据
            try:
                # 回调下载成功列表
                logging.info(
                    "用户ID: {}, 开始回调 {} 结果, 共 {} 条数据...".format(
                        user_id, conf.handle_result_url, len(call_res_dict)))
                call_results_status = call_url_func(user_id, conf.handle_result_url, data_json={
                    'userId': user_id,
                    'content': call_res_dict
                })

                if not call_results_status:
                    # 回调失败, 保存结果
                    logging.error("用户ID: {}, 结果列表回调失败, 保存结果至oss!".format(user_id))
                    oss_bucket.put_object(
                        "face_cluster_call_error_data/{}/call_result_error.pkl".format(user_id),
                        pickle.dumps({
                            "handle_result_url": conf.handle_result_url,
                            "user_id": user_id,
                            "call_res_dict": call_res_dict,
                        }))
            except Exception as e:
                logging.exception(e)
                time.sleep(2)
                continue
            time.sleep(2)


class ImageProcessingThread(threading.Thread):  # 继承父类threading.Thread
    def __init__(self, thread_name):
        threading.Thread.__init__(self)
        self.thread_name = thread_name

    def pr_log(self, content, style='info'):
        log_content = "线程名: {}, {}".format(self.thread_name, content)
        if style == 'info':
            logging.info(log_content)
        elif style == "error":
            logging.error(log_content)
        elif style == 'exception':
            logging.exception(content)

    def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
        self.pr_log("开始加载局部模型...")
        aesthetic_model = image_quality_assessment_interface.QualityAssessmentModel(
            model_path='./models/weights_mobilenet_aesthetic_0.07.hdf5')
        technical_model = image_quality_assessment_interface.QualityAssessmentModel(
            model_path='./models/weights_mobilenet_technical_0.11.hdf5')
        fr_arcface = face_recognition_interface.FaceRecognitionWithArcFace()
        fe_detectrion = face_emotion_interface.FaceEmotionKeras()  # 表情检测模型, 不能跨线程
        while True:
            params_count = r_object.llen_content(conf.res_image_making_name)
            try:
                self.pr_log("开始处理图片, 剩余数据: {} 条".format(params_count))
                params = r_object.rpop_content(conf.redis_image_queue_name)
                if params:
                    params = pickle.loads(params)
                    user_id = params.get("user_id")
                    media_id = params.get("media_id")
                    image_url = params.get('image_url')
                    file_id = params.get('file_id')
                    callback_url = params.get('callback_url')

                    image_name = os.path.basename(image_url)
                    img_path = os.path.join(conf.tmp_image_dir, image_name)
                    try:
                        start_time = time.time()
                        urlretrieve(image_url, img_path)
                        image_split = os.path.splitext(image_name)
                        if image_split[-1].lower == 'heic':
                            new_img_path = os.path.join(conf.tmp_image_dir, "{}.jpg".format(image_split[0]))
                            os.system("tifig -v -p {} {}".format(img_path, new_img_path))
                            if os.path.isfile(img_path) and os.path.isfile(new_img_path):
                                os.remove(img_path)
                                img_path = new_img_path
                        assessment_img = np.asarray(
                            keras.preprocessing.image.load_img(img_path, target_size=(224, 224)))
                        aesthetic_value = aesthetic_model.get_res(assessment_img)
                        technical_value = technical_model.get_res(assessment_img)
                        aesthetic_value = aesthetic_value * 0.8 + technical_value * 0.2

                        image = cv2.imread(img_path)
                        tags = oi_5000_model.get_tag(img_path) + \
                               ml_1000_model.get_tag(image) + \
                               ml_11166_model.get_tag(image)

                        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        im_height, im_width = image.shape[:2]
                        f = min((4096. / max(image.shape[0], image.shape[1])), 1.0)
                        image_r = cv2.resize(image, (int(image.shape[1] * f), int(image.shape[0] * f)))

                        # image_np = cv2.cvtColor(image_r, cv2.COLOR_BGR2RGB)
                        image_np_expanded = np.expand_dims(image_r, axis=0)

                        (fd_boxes_, fd_scores_) = fd_ssd_detection.detect_face(image_np_expanded)
                        face_count = 0
                        for i in range(fd_boxes_[0].shape[0]):
                            if fd_scores_[0][i] < 0.7:
                                break
                            ymin, xmin, ymax, xmax = fd_boxes_[0][i]
                            add_y_border = (ymax - ymin) * 0.1
                            add_x_border = (xmax - xmin) * 0.1
                            xmin_, xmax_ = max(0, xmin - add_x_border), min(1, xmax + add_x_border)
                            ymin_, ymax_ = max(0, ymin - add_y_border), min(1, ymax + add_y_border)
                            left, right, top, bottom = map(
                                int, (xmin_ * im_width, xmax_ * im_width, ymin_ * im_height, ymax_ * im_height))

                            face_image = image_np[top:bottom, left:right, :]
                            if max(face_image.shape[:2]) < 56.:
                                continue
                            face_image_resize, im_scale = image_resize(face_image)
                            mtcnn_res = fd_mtcnn_detection.detect_face(face_image_resize)
                            if not mtcnn_res:
                                continue
                            mtcnn_scare = mtcnn_res['confidence']
                            if mtcnn_scare < 0.96:
                                continue
                            mtcnn_box = mtcnn_res['box']
                            if max(mtcnn_box[2:]) < 50.:
                                continue
                            mtcnn_points = mtcnn_res['keypoints']
                            mtcnn_points = np.asarray([
                                [mtcnn_points['left_eye'][0] / im_scale + left,
                                 mtcnn_points['left_eye'][1] / im_scale + top],
                                [mtcnn_points['right_eye'][0] / im_scale + left,
                                 mtcnn_points['right_eye'][1] / im_scale + top],
                                [mtcnn_points['nose'][0] / im_scale + left, mtcnn_points['nose'][1] / im_scale + top],
                                [mtcnn_points['mouth_left'][0] / im_scale + left,
                                 mtcnn_points['mouth_left'][1] / im_scale + top],
                                [mtcnn_points['mouth_right'][0] / im_scale + left,
                                 mtcnn_points['mouth_right'][1] / im_scale + top],
                            ])
                            warped = image_tools.preprocess(image_np, [112, 112], bbox=mtcnn_box, landmark=mtcnn_points)
                            face_image_name = os.path.join(
                                conf.tmp_image_dir, "{}_{}.jpg".format(media_id, i))
                            cv2.imwrite(face_image_name, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                            oss_face_image_name = "face_cluster_data/{}/face_images/{}_{}.jpg".format(
                                user_id, media_id, i)
                            oss_bucket.put_object_from_file(oss_face_image_name, face_image_name)
                            if os.path.isfile(face_image_name):
                                os.remove(face_image_name)

                            emotion_label_arg = fe_detectrion.detection_emotion(warped)

                            warped = np.transpose(warped, (2, 0, 1))
                            emb = fr_arcface.get_feature(warped)

                            redis_user_key = conf.redis_face_info_name.format(user_id)
                            r_object.lpush_content(redis_user_key, pickle.dumps({
                                "face_id": "{}_{}".format(media_id, i),
                                "face_box": [left, top, right - left, bottom - top],
                                "face_feature": emb,
                                "emotionStr": emotion_label_arg,
                            }))
                            r_object.lpush_content(conf.redis_face_info_key_list, redis_user_key)
                            # if not r_object.sismember_content(conf.redis_face_info_key_set, redis_user_key):  # 不在队列里
                            #     r_object.sadd_content(conf.redis_face_info_key_set, redis_user_key)

                            face_count += 1
                        data_json = {
                            'mediaId': media_id,
                            'fileId': file_id,
                            'tag': str(tags),
                            'filePath': image_url,
                            # 'exponent': str({"aesthetic": aesthetic_value, "technical": technical_value}),
                            'exponent': aesthetic_value,
                            'identity': str({"isIDCard": 0}),
                            'existFace': min(face_count, 127),
                        }

                        call_res = requests.post(callback_url, json=data_json)
                        self.pr_log("返回代码: {}, 返回内容: {}".format(call_res.status_code, call_res.text))
                        self.pr_log("{} 处理成功, 耗时: {}".format(media_id, time.time() - start_time))
                    except Exception as e:
                        self.pr_log("{} 处理失败!".format(img_path), style='error')
                        self.pr_log(e, style='exception')
                        continue
            except Exception as e:
                self.pr_log(e, style='exception')
                continue
            finally:
                reboot_status = r_object.get_content(local_ip)
                reboot_code = str(reboot_status)
                if reboot_code == '1':
                    self.pr_log('发现服务需要重启, 重启代码: {}'.format(reboot_code))
                    return
                time.sleep(9 / max(1, params_count))


if __name__ == '__main__':
    # 创建日志文件
    util.makedirs(conf.log_dir)
    py_file_name = os.path.basename(__file__).split('.')[0]
    log_file = os.path.join(conf.log_dir, "{}.log".format(py_file_name))
    util.init_logging(log_file, log_filelevel=logging.INFO, log_streamlevel=logging.INFO, daily=False)
    logging.info("开始启动程序...")
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    local_ip = util.get_local_ip()  # 获取ip
    env_code = conf.env_manage.get(local_ip, 0)
    logging.info("获取运行环境代码为: {}(0: 测试服, 1: 正式服), 开始连接oss和redis...".format(env_code))
    account_conf = conf.AccountConf(env_code=env_code)
    # 连接oss
    oss_bucket = connects.ConnectALiYunOSS(
        account_conf.accessKeyId, account_conf.accessKeySecret, account_conf.endpoint, account_conf.bucket)
    # 连接redis
    r_object = connects.ConnectRedis(
        account_conf.res_host, account_conf.res_port, account_conf.res_decode_responses, account_conf.res_password)
    # 设置脚本重启代码
    r_object.set_content(local_ip, "0")

    logging.info("加载全局模型...")
    oi_5000_model = image_making_interface.ImageMakingWithOpenImage()
    ml_1000_model = image_making_interface.ImageMakingWithTencent(
        model_path='./models/tencent_1000.pb', label_path='./data/ml_label_1000.txt')
    ml_11166_model = image_making_interface.ImageMakingWithTencent(
        model_path='./models/tencent_11166.pb', label_path='./data/ml_label_11166.txt')
    fd_ssd_detection = face_detection_interface.FaceDetectionWithSSDMobilenet()
    fd_mtcnn_detection = face_detection_interface.FaceDetectionWithMtcnnTF(steps_threshold=[0.6, 0.7, 0.8])

    logging.info("即将开启的线程数: {}".format(conf.thread_num))
    # 创建线程并开始线程
    for i in range(conf.thread_num):
        ImageProcessingThread("Thread_{}".format(i)).start()

    while True:
        active_thread_count = threading.active_count()
        if active_thread_count == 1:
            logging.info('所有线程已停止, 等待重启...')
            break
        time.sleep(3)