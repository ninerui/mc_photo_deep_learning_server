# 设置运行环境, 1表示正式环境, 0表示测试环境
env_manage = {
    # '172.16.181.140': 1,  # 118.31.225.209
    '172.18.194.177': 0,  # 47.112.115.127
    '172.16.172.127': 0,
    '172.16.172.130': 0,

    '172.16.178.202': 1,  # 47.111.147.203, 深度学习01
    '172.16.178.203': 1,  # 47.111.159.131, 深度学习02
    '172.16.178.210': 1,  # 47.111.154.70, 深度学习03
}

image_process_thread_num_dict = {  # 打标线程数
    '172.16.172.127': 3,

    '172.16.178.202': 3,
    '172.16.178.203': 3,
}
face_cluster_thread_num_dict = {  # 人脸聚类线程数
    '172.16.172.130': 2,

    '172.16.178.210': 3,
}
wonderful_gen_thread_num_dict = {  # 精彩生成线程数
    '172.16.172.130': 2,

    '172.16.178.210': 2,
}
log_dir = './log'
tmp_image_dir = '/data/tmp_image_dir'
tmp_youkan_image_dir = "/data/tmp_youkan_image_dir"

# redis_image_queue_name = "mc_image_queue_list"
redis_face_info_name = "mc_face_info_list-{}"
redis_face_info_key_set = "mc_face_info_key_set"
redis_face_info_key_list = "mc_face_info_key_list"

redis_image_making_list_name = "mc_image_making_list"
redis_image_making_set_name = "mc_image_making_set"
redis_image_making_error_name = "mc_image_making_error"
redis_wonderful_gen_name = "mc_wonderful_gen_list"

# res_face_cluster_name = "mc_face_cluster_list"
# res_image_making_name = "mc_image_making_list"

handle_result_url_dict = {
    '0': [
        "http://172.16.181.135:8081/ai/updateFaceId"
    ],
    '1': [
        "http://172.16.107.2:8081/ai/updateFaceId",
        "http://172.16.178.209:8081/ai/updateFaceId"
    ]
}
wonderful_callback_url_dict = {
    '0': [
        "http://172.16.181.135:8081/ai/receiveWonderfulData"
    ],
    '1': [
        "http://172.16.107.2:8081/ai/receiveWonderfulData",
        "http://172.16.178.209:8081/ai/receiveWonderfulData"
    ]
}


# handle_result_url = "http://172.16.181.135:8081/ai/updateFaceId"
#
# wonderful_callback_url = 'http://172.16.181.135:8081/ai/receiveWonderfulData'


class AccountConf:
    def __init__(self, env_code):
        assert env_code in [0, 1]
        if env_code == 1:  # 正式环境
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
        else:  # 测试环境
            # oss对象
            self.accessKeyId = 'LTAIRje9cQipM55x'
            self.accessKeySecret = 'U0oKHXSxsBavNnkzrGcWkpaDlHs9yH'
            self.endpoint = 'http://oss-cn-hangzhou-internal.aliyuncs.com'
            # self.endpoint = 'http://oss-cn-shenzhen.aliyuncs.com'
            self.bucket = 'mc-photo-face-ai'
            self.host = 'oss-cn-hangzhou-internal.aliyuncs.com'
            self.res_host = 'r-bp1s3ywv637g7qrzfy.redis.rds.aliyuncs.com'
            self.res_port = 6379
            self.res_decode_responses = True
            self.res_password = "Aszs2019"


class WritingAccountConf(object):
    def __init__(self, env_code):
        assert env_code in [0, 1]  # 0为测试服, 1为正式环境
        # oss对象
        self.accessKeyId = ['LTAIRje9cQipM55x', 'LTAIPXCD6YqmhHgz'][env_code]
        self.accessKeySecret = ['U0oKHXSxsBavNnkzrGcWkpaDlHs9yH', 'XMnLVGCkJTGtDfz7xMZVjgVZAnaSPD'][env_code]
        self.endpoint = [
            'http://oss-cn-hangzhou-internal.aliyuncs.com',
            'http://oss-cn-hangzhou-internal.aliyuncs.com'][env_code]
        self.bucket = ['mc-photo-writing', 'mc-photo'][env_code]
        self.host = ['oss-cn-hangzhou-internal.aliyuncs.com', ][env_code]


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
