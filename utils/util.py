import os
import socket
import logging
import logging.handlers


def makedirs(path):
    """创建目录"""
    if not os.path.exists(path):
        os.makedirs(path)


def removefile(path):
    if os.path.isfile(path):
        os.remove(path)


def get_local_ip():
    # 获取本机ip
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    return str(local_ip)


def init_logging(log_filename,
                 log_filelevel=logging.DEBUG,
                 log_errorlevel=logging.ERROR,
                 log_streamlevel=logging.DEBUG,
                 daily=True,
                 datefmt='%H:%M:%S'):
    """日志初始化"""
    logging.basicConfig(
        # stream=sys.stdout,
        level=log_streamlevel,
        format='%(asctime)s.%(msecs)03d %(levelname)s : %(message)s',
        datefmt=datefmt)
    # 日志文件设置
    if log_filename:
        if os.path.split(log_filename)[0]:
            makedirs(os.path.split(log_filename)[0])
        if daily:
            file_handler = logging.handlers.TimedRotatingFileHandler(log_filename, when='MIDNIGHT')
        else:
            file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(log_filelevel)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s : %(message)s', datefmt=datefmt)
        )
        logging.getLogger().addHandler(file_handler)

        if log_errorlevel:
            if daily:
                errorfile_handler = logging.handlers.TimedRotatingFileHandler(log_filename + ".ERROR", when='MIDNIGHT')
            else:
                errorfile_handler = logging.FileHandler(log_filename + ".ERROR")
            errorfile_handler.setLevel(log_errorlevel)
            errorfile_handler.setFormatter(
                logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s : %(message)s', datefmt=datefmt)
            )
            logging.getLogger().addHandler(errorfile_handler)
