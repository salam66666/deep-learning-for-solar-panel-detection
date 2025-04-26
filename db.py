# db.py
from mysql.connector import pooling

dbconfig = {
    "database": "yolo_web",
    "user": "root",
    # # 本地
    "password": "123456",
    "host": "localhost",
    # # 线上
    # "password": "123456qwer",
    # "host": "localhost"
}

# 初始化连接池
pool_name = "mypool"
pool_size = 16

cnxpool = pooling.MySQLConnectionPool(pool_name=pool_name,
                                      pool_size=pool_size,
                                      pool_reset_session=True,
                                      **dbconfig)
