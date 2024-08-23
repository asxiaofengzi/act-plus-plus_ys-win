from dynamixel_client import DynamixelClient  # 从dynamixel_client模块导入DynamixelClient类

# 创建DynamixelClient实例，控制ID为1和2的舵机，指定端口为'/dev/ttyDXL_wheels'，并设置为懒连接模式
client = DynamixelClient([1, 2], port='/dev/ttyDXL_wheels', lazy_connect=True)

# 读取舵机的当前位置、速度和电流，并打印结果
print(client.read_pos_vel_cur())
