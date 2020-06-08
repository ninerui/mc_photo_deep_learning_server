# gunicorn 运行模块名:应用名 -c 配置文件
# gunicorn deep_learning_server_api:app -c gunicorn.conf

# 并行工作线程数
workers = 4
# 监听内网端口5000【按需要更改】
bind = '127.0.0.1:8081'
# 设置守护进程【关闭连接时，程序仍在运行】
daemon = True
# 设置超时时间120s，默认为30s。按自己的需求进行设置
timeout = 120
# 设置访问日志和错误信息日志路径
accesslog = './log/acess.log'
errorlog = './log/error.log'
# access_log_format = '%(h)s %(l)s %(u)s %(t)s'

"""
debug:调试级别，记录的信息最多；
info:普通级别；
warning:警告消息；
error:错误消息；
critical:严重错误消息；
"""
# loglevel = 'error'


# logger_class = 'STRING'  # 选择处理日志的方法
