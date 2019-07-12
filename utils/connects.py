import pickle

import oss2
import redis


class ConnectRedis:
    def __init__(self, res_host, res_port, res_decode_responses, res_password):
        r_pool = redis.ConnectionPool(
            host=res_host, port=res_port, decode_responses=res_decode_responses, password=res_password)
        self.r_object = redis.Redis(connection_pool=r_pool)

    def get_content(self, key):
        return self.r_object.get(key)

    def set_content(self, key, data):
        self.r_object.set(key, data)

    def llen_content(self, key):
        return self.r_object.llen(key)

    def rpop_content(self, key):
        return self.r_object.rpop(key)

    def lpush_content(self, key, value):
        self.r_object.lpush(key, value)


class ConnectALiYunOSS:
    def __init__(self, accessKeyId, accessKeySecret, endpoint, bucket):
        oss_auth = oss2.Auth(accessKeyId, accessKeySecret)
        self.oss_bucket = oss2.Bucket(oss_auth, endpoint, bucket)

    def put_object(self, key, data):
        self.oss_bucket.put_object(key, data)

    def get_object(self, key):
        return self.oss_bucket.get_object(key)

    def object_exists(self, key):
        return self.oss_bucket.object_exists(key)

    def delete_object(self, key):
        self.oss_bucket.delete_object(key)

    def put_object_from_file(self, key, data):
        self.oss_bucket.put_object_from_file(key, data)

    def get_set_object(self, key, data):
        oss_exist = self.object_exists(key)
        if not oss_exist:
            self.put_object(key, pickle.dumps(data))
