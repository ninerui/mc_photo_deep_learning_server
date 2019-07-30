# 设置运行环境, 1表示正式环境, 0表示测试环境
env_manage = {
    # '172.16.181.140': 1,  # 118.31.225.209
    '172.18.194.177': 0,  # 47.112.115.127
    # '172.16.172.124': 1,  # 47.99.176.149
    '172.16.178.202': 1,  # 47.111.147.203
    '172.16.178.203': 1,  # 47.111.147.95
}
log_dir = './log'
tmp_image_dir = '../tmp_image_dir'

image_process_thread_num = 3
face_cluster_thread_num = 1
wonderful_gen_thread_num = 1

thread_num = 3

redis_image_queue_name = "mc_image_queue_list"
redis_face_info_name = "mc_face_info_list-{}"
redis_face_info_key_set = "mc_face_info_key_set"
redis_face_info_key_list = "mc_face_info_key_list"

res_face_cluster_name = "mc_face_cluster_list"
res_image_making_name = "mc_image_making_list"
res_wonderful_gen_name = "mc_wonderful_gen_list"

handle_result_url = "http://47.112.110.202:8081/ai/updateFaceId"


class AccountConf:
    def __init__(self, env_code):
        assert env_code in [0, 1]
        if env_code == 1:
            # oss对象
            self.accessKeyId = 'LTAI3MSuaxizYX7d'
            self.accessKeySecret = 'YEkkxK1nOeZgVJMl3UeVSdkFavjemu'
            self.endpoint = 'http://oss-cn-hangzhou-internal.aliyuncs.com'
            self.bucket = 'mc-photo-ai'
            self.host = 'oss-cn-hangzhou-internal.aliyuncs.com'
            # redis addr
            self.res_host = 'r-bp10d6bqy8w5t5qz4f.redis.rds.aliyuncs.com'
            self.res_port = 6379
            self.res_decode_responses = True
            self.res_password = "McPhotos$2019"
        else:
            # oss对象
            self.accessKeyId = 'LTAIu0Zf15FLBq4R'
            self.accessKeySecret = 'n9zAUBrXhf2nFilRowvPqkNOrw8u75'
            # endpoint = 'http://oss-cn-shenzhen-internal.aliyuncs.com'
            self.endpoint = 'http://oss-cn-shenzhen.aliyuncs.com'
            self.bucket = 'mc-photo-ai-face'
            self.host = 'oss-cn-shenzhen-internal.aliyuncs.com'
            # redis addr
            self.res_host = 'localhost'
            self.res_port = 6379
            self.res_decode_responses = True
            self.res_password = ""


# 返回代码参数
status_code = {
    "00": {
        "data": "",
        "message": "Success",
        "code": 0,
    },
    "29": {
        "data": "",
        "message": "no userId",
        "code": 29
    },
    "30": {
        "data": "",
        "message": "no baseUrl",
        "code": 30
    },
    "31": {
        "data": "",
        "message": "no callbackUr",
        "code": 31
    },
    "32": {
        "data": "",
        "message": "no handleResultUrl",
        "code": 32
    },
    "33": {
        "data": "",
        "message": "no content",
        "code": 33
    },
    "36": {
        "data": "",
        "message": "content parser error",
        "code": 36
    },
    "34": {
        "data": "",
        "message": "userId running",
        "code": 34
    },
    "28": {
        "data": "",
        "message": "no use POST request",
        "code": 28
    },
    "35": {
        "data": "",
        "message": "os error",
        "code": 35
    },
    "37": {
        "data": "",
        "message": "save redis error",
        "code": 37
    },
}
