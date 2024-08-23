"""Communication using the DynamixelSDK."""
##This is based off of the dynamixel SDK
import atexit  # 导入atexit模块，用于注册程序退出时的清理函数
import logging  # 导入logging模块，用于记录日志
import time  # 导入time模块，用于时间相关操作
from typing import Optional, Sequence, Union, Tuple  # 导入类型提示模块

import numpy as np  # 导入NumPy模块，用于数组操作

PROTOCOL_VERSION = 2.0  # 定义Dynamixel协议版本

# 以下地址假设使用的是XH系列电机
ADDR_TORQUE_ENABLE = 64  # 扭矩使能地址
ADDR_GOAL_POSITION = 116  # 目标位置地址
ADDR_PRESENT_POSITION = 132  # 当前位置信息地址
ADDR_PRESENT_VELOCITY = 128  # 当前速度信息地址
ADDR_PRESENT_CURRENT = 126  # 当前电流信息地址
ADDR_PRESENT_POS_VEL_CUR = 126  # 当前位置信息、速度信息和电流信息的起始地址

# 数据字节长度
LEN_PRESENT_POSITION = 4  # 当前位置信息的字节长度
LEN_PRESENT_VELOCITY = 4  # 当前速度信息的字节长度
LEN_PRESENT_CURRENT = 2  # 当前电流信息的字节长度
LEN_PRESENT_POS_VEL_CUR = 10  # 当前位置信息、速度信息和电流信息的总字节长度
LEN_GOAL_POSITION = 4  # 目标位置的字节长度

DEFAULT_POS_SCALE = 2.0 * np.pi / 4096  # 默认位置缩放比例（0.088度）
# 参见 http://emanual.robotis.com/docs/en/dxl/x/xh430-v210/#goal-velocity
DEFAULT_VEL_SCALE = 0.229 * 2.0 * np.pi / 60.0  # 默认速度缩放比例（0.229转每分钟）
DEFAULT_CUR_SCALE = 1.34  # 默认电流缩放比例

def dynamixel_cleanup_handler():
    """Cleanup function to ensure Dynamixels are disconnected properly."""
    open_clients = list(DynamixelClient.OPEN_CLIENTS)  # 获取所有打开的客户端
    for open_client in open_clients:
        if open_client.port_handler.is_using:
            logging.warning('Forcing client to close.')  # 如果端口正在使用，记录警告日志
        open_client.port_handler.is_using = False  # 设置端口不再使用
        open_client.disconnect()  # 断开客户端连接

def signed_to_unsigned(value: int, size: int) -> int:
    """Converts the given value to its unsigned representation."""
    if value < 0:
        bit_size = 8 * size  # 计算位大小
        max_value = (1 << bit_size) - 1  # 计算最大值
        value = max_value + value  # 转换为无符号表示
    return value

def unsigned_to_signed(value: int, size: int) -> int:
    """Converts the given value from its unsigned representation."""
    bit_size = 8 * size  # 计算位大小
    if (value & (1 << (bit_size - 1))) != 0:
        value = -((1 << bit_size) - value)  # 转换为有符号表示
    return value

class DynamixelClient:
    """Client for communicating with Dynamixel motors.

    NOTE: This only supports Protocol 2.
    """

    # The currently open clients.
    OPEN_CLIENTS = set()  # 存储当前打开的客户端

    def __init__(self,
                 motor_ids: Sequence[int],
                 port: str = '/dev/ttyUSB0',
                 baudrate: int = 1000000,
                 lazy_connect: bool = False,
                 pos_scale: Optional[float] = None,
                 vel_scale: Optional[float] = None,
                 cur_scale: Optional[float] = None):
        """Initializes a new client.

        Args:
            motor_ids: All motor IDs being used by the client.
            port: The Dynamixel device to talk to. e.g.
                - Linux: /dev/ttyUSB0
                - Mac: /dev/tty.usbserial-*
                - Windows: COM1
            baudrate: The Dynamixel baudrate to communicate with.
            lazy_connect: If True, automatically connects when calling a method
                that requires a connection, if not already connected.
            pos_scale: The scaling factor for the positions. This is
                motor-dependent. If not provided, uses the default scale.
            vel_scale: The scaling factor for the velocities. This is
                motor-dependent. If not provided uses the default scale.
            cur_scale: The scaling factor for the currents. This is
                motor-dependent. If not provided uses the default scale.
        """
        import dynamixel_sdk  # 导入Dynamixel SDK
        self.dxl = dynamixel_sdk  # 将Dynamixel SDK赋值给实例变量

        self.motor_ids = list(motor_ids)  # 将传入的电机ID列表转换为列表并赋值给实例变量
        self.port_name = port  # 保存端口名称
        self.baudrate = baudrate  # 保存波特率
        self.lazy_connect = lazy_connect  # 保存是否懒连接的标志
        
        self.port_handler = self.dxl.PortHandler(port)  # 创建端口处理器对象
        self.packet_handler = self.dxl.PacketHandler(PROTOCOL_VERSION)  # 创建数据包处理器对象
        
        # 创建位置、速度和电流读取器对象
        self._pos_vel_cur_reader = DynamixelPosVelCurReader(
            self,
            self.motor_ids,
            pos_scale=pos_scale if pos_scale is not None else DEFAULT_POS_SCALE,  # 使用传入的或默认的位置缩放比例
            vel_scale=vel_scale if vel_scale is not None else DEFAULT_VEL_SCALE,  # 使用传入的或默认的速度缩放比例
            cur_scale=cur_scale if cur_scale is not None else DEFAULT_CUR_SCALE,  # 使用传入的或默认的电流缩放比例
        )
        
        # 创建位置读取器对象
        self._pos_reader = DynamixelPosReader(
            self,
            self.motor_ids,
            pos_scale=pos_scale if pos_scale is not None else DEFAULT_POS_SCALE,  # 使用传入的或默认的位置缩放比例
            vel_scale=vel_scale if vel_scale is not None else DEFAULT_VEL_SCALE,  # 使用传入的或默认的速度缩放比例
            cur_scale=cur_scale if cur_scale is not None else DEFAULT_CUR_SCALE,  # 使用传入的或默认的电流缩放比例
        )
        
        # 创建速度读取器对象
        self._vel_reader = DynamixelVelReader(
            self,
            self.motor_ids,
            pos_scale=pos_scale if pos_scale is not None else DEFAULT_POS_SCALE,  # 使用传入的或默认的位置缩放比例
            vel_scale=vel_scale if vel_scale is not None else DEFAULT_VEL_SCALE,  # 使用传入的或默认的速度缩放比例
            cur_scale=cur_scale if cur_scale is not None else DEFAULT_CUR_SCALE,  # 使用传入的或默认的电流缩放比例
        )
        
        # 创建电流读取器对象
        self._cur_reader = DynamixelCurReader(
            self,
            self.motor_ids,
            pos_scale=pos_scale if pos_scale is not None else DEFAULT_POS_SCALE,  # 使用传入的或默认的位置缩放比例
            vel_scale=vel_scale if vel_scale is not None else DEFAULT_VEL_SCALE,  # 使用传入的或默认的速度缩放比例
            cur_scale=cur_scale if cur_scale is not None else DEFAULT_CUR_SCALE,  # 使用传入的或默认的电流缩放比例
        )
        
        self._sync_writers = {}  # 初始化同步写入器字典
        
        self.OPEN_CLIENTS.add(self)  # 将当前客户端实例添加到打开的客户端集合中
        
        @property
        def is_connected(self) -> bool:
            return self.port_handler.is_open  # 返回端口是否打开的状态
        
        def connect(self):
            """Connects to the Dynamixel motors.
        
            NOTE: This should be called after all DynamixelClients on the same
                process are created.
            """
            assert not self.is_connected, 'Client is already connected.'  # 断言客户端未连接
        
            if self.port_handler.openPort():
                logging.info('Succeeded to open port: %s', self.port_name)  # 记录成功打开端口的日志
            else:
                raise OSError(
                    ('Failed to open port at {} (Check that the device is powered '
                     'on and connected to your computer).').format(self.port_name))  # 抛出打开端口失败的异常
        
            if self.port_handler.setBaudRate(self.baudrate):
                logging.info('Succeeded to set baudrate to %d', self.baudrate)  # 记录成功设置波特率的日志
            else:
                raise OSError(
                    ('Failed to set the baudrate to {} (Ensure that the device was '
                     'configured for this baudrate).').format(self.baudrate))  # 抛出设置波特率失败的异常
        
            # Start with all motors enabled.  NO, I want to set settings before enabled
            #self.set_torque_enabled(self.motor_ids, True)
        
        def disconnect(self):
            """Disconnects from the Dynamixel device."""
            if not self.is_connected:
                return  # 如果未连接，直接返回
            if self.port_handler.is_using:
                logging.error('Port handler in use; cannot disconnect.')  # 记录端口正在使用，无法断开的错误日志
                return
            # Ensure motors are disabled at the end.
            self.set_torque_enabled(self.motor_ids, False, retries=0)  # 确保在断开连接前禁用电机
            self.port_handler.closePort()  # 关闭端口
            if self in self.OPEN_CLIENTS:
                self.OPEN_CLIENTS.remove(self)  # 从打开的客户端集合中移除当前实例

    def set_torque_enabled(self,
                           motor_ids: Sequence[int],
                           enabled: bool,
                           retries: int = -1,
                           retry_interval: float = 0.25):
        """Sets whether torque is enabled for the motors.
    
        Args:
            motor_ids: The motor IDs to configure.
            enabled: Whether to engage or disengage the motors.
            retries: The number of times to retry. If this is <0, will retry
                forever.
            retry_interval: The number of seconds to wait between retries.
        """
        remaining_ids = list(motor_ids)  # 将电机ID列表转换为列表
        while remaining_ids:  # 当仍有未成功设置的电机ID时
            remaining_ids = self.write_byte(
                remaining_ids,
                int(enabled),
                ADDR_TORQUE_ENABLE,
            )  # 尝试写入扭矩使能值
            if remaining_ids:  # 如果仍有未成功设置的电机ID
                logging.error('Could not set torque %s for IDs: %s',
                              'enabled' if enabled else 'disabled',
                              str(remaining_ids))  # 记录错误日志
            if retries == 0:  # 如果重试次数为0
                break  # 退出循环
            time.sleep(retry_interval)  # 等待一段时间后重试
            retries -= 1  # 减少重试次数
    
    def read_pos_vel_cur(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the current positions and velocities."""
        return self._pos_vel_cur_reader.read()  # 调用位置、速度和电流读取器的read方法
    
    def read_pos(self) -> np.ndarray:
        """Returns the current positions."""
        return self._pos_reader.read()  # 调用位置读取器的read方法
    
    def read_vel(self) -> np.ndarray:
        """Returns the current velocities."""
        return self._vel_reader.read()  # 调用速度读取器的read方法
    
    def read_cur(self) -> np.ndarray:
        """Returns the current currents."""
        return self._cur_reader.read()  # 调用电流读取器的read方法
    
    def write_desired_pos(self, motor_ids: Sequence[int], positions: np.ndarray):
        """Writes the given desired positions.
    
        Args:
            motor_ids: The motor IDs to write to.
            positions: The joint angles in radians to write.
        """
        assert len(motor_ids) == len(positions)  # 确保电机ID和位置的数量相同
    
        # Convert to Dynamixel position space.
        positions = positions / self._pos_vel_cur_reader.pos_scale  # 将位置转换为Dynamixel位置空间
        self.sync_write(motor_ids, positions, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)  # 同步写入目标位置
    
    def write_byte(self,
                   motor_ids: Sequence[int],
                   value: int,
                   address: int) -> Sequence[int]:
        """Writes a value to the motors.
    
        Args:
            motor_ids: The motor IDs to write to.
            value: The value to write to the control table.
            address: The control table address to write to.
    
        Returns:
            A list of IDs that were unsuccessful.
        """
        self.check_connected()  # 检查是否已连接
        errored_ids = []  # 初始化错误ID列表
        for motor_id in motor_ids:  # 遍历每个电机ID
            comm_result, dxl_error = self.packet_handler.write1ByteTxRx(
                self.port_handler, motor_id, address, value)  # 写入一个字节的数据
            success = self.handle_packet_result(
                comm_result, dxl_error, motor_id, context='write_byte')  # 处理数据包结果
            if not success:  # 如果写入失败
                errored_ids.append(motor_id)  # 将电机ID添加到错误ID列表中
        return errored_ids  # 返回错误ID列表

    def sync_write(self, motor_ids: Sequence[int],
                   values: Sequence[Union[int, float]], address: int,
                   size: int):
        """Writes values to a group of motors.

        Args:
            motor_ids: The motor IDs to write to.
            values: The values to write.
            address: The control table address to write to.
            size: The size of the control table value being written to.
        """
        self.check_connected()  # 确保已经连接到设备
        key = (address, size)  # 创建地址和大小的键
        if key not in self._sync_writers:  # 如果同步写入器字典中没有这个键
            self._sync_writers[key] = self.dxl.GroupSyncWrite(
                self.port_handler, self.packet_handler, address, size)  # 创建一个新的GroupSyncWrite对象并存储在字典中
        sync_writer = self._sync_writers[key]  # 获取同步写入器对象

        errored_ids = []  # 初始化错误ID列表
        for motor_id, desired_pos in zip(motor_ids, values):  # 遍历电机ID和对应的目标位置
            value = signed_to_unsigned(int(desired_pos), size=size)  # 将目标位置转换为无符号整数
            value = value.to_bytes(size, byteorder='little')  # 将无符号整数转换为字节
            success = sync_writer.addParam(motor_id, value)  # 添加参数到同步写入器
            if not success:  # 如果添加参数失败
                errored_ids.append(motor_id)  # 将电机ID添加到错误ID列表中

        if errored_ids:  # 如果有错误ID
            logging.error('Sync write failed for: %s', str(errored_ids))  # 记录错误日志

        comm_result = sync_writer.txPacket()  # 发送同步写入数据包
        self.handle_packet_result(comm_result, context='sync_write')  # 处理数据包结果

        sync_writer.clearParam()  # 清除同步写入器的参数

    def check_connected(self):
        """Ensures the robot is connected."""
        if self.lazy_connect and not self.is_connected:  # 如果是懒连接且未连接
            self.connect()  # 进行连接
        if not self.is_connected:  # 如果仍未连接
            raise OSError('Must call connect() first.')  # 抛出异常

    def handle_packet_result(self,
                             comm_result: int,
                             dxl_error: Optional[int] = None,
                             dxl_id: Optional[int] = None,
                             context: Optional[str] = None):
        """Handles the result from a communication request."""
        error_message = None  # 初始化错误信息
        if comm_result != self.dxl.COMM_SUCCESS:  # 如果通信结果不成功
            error_message = self.packet_handler.getTxRxResult(comm_result)  # 获取通信结果的错误信息
        elif dxl_error is not None:  # 如果有Dynamixel错误
            error_message = self.packet_handler.getRxPacketError(dxl_error)  # 获取Dynamixel错误信息
        if error_message:  # 如果有错误信息
            if dxl_id is not None:  # 如果有电机ID
                error_message = '[Motor ID: {}] {}'.format(dxl_id, error_message)  # 添加电机ID到错误信息
            if context is not None:  # 如果有上下文信息
                error_message = '> {}: {}'.format(context, error_message)  # 添加上下文信息到错误信息
            logging.error(error_message)  # 记录错误日志
            return False  # 返回False表示处理失败
        return True  # 返回True表示处理成功

    def convert_to_unsigned(self, value: int, size: int) -> int:
        """Converts the given value to its unsigned representation."""
        if value < 0:  # 如果值为负数
            max_value = (1 << (8 * size)) - 1  # 计算最大值
            value = max_value + value  # 转换为无符号表示
        return value  # 返回无符号值

    def __enter__(self):
        """Enables use as a context manager."""
        if not self.is_connected:  # 如果未连接
            self.connect()  # 进行连接
        return self  # 返回自身实例

    def __exit__(self, *args):
        """Enables use as a context manager."""
        self.disconnect()  # 断开连接

    def __del__(self):
        """Automatically disconnect on destruction."""
        self.disconnect()  # 断开连接


class DynamixelReader:
    """Reads data from Dynamixel motors.

    This wraps a GroupBulkRead from the DynamixelSDK.
    """

    def __init__(self, client: DynamixelClient, motor_ids: Sequence[int],
                 address: int, size: int):
        """Initializes a new reader."""
        self.client = client  # 保存Dynamixel客户端实例
        self.motor_ids = motor_ids  # 保存电机ID列表
        self.address = address  # 保存读取数据的起始地址
        self.size = size  # 保存读取数据的字节大小
        self._initialize_data()  # 初始化缓存数据

        # 创建GroupBulkRead对象，用于批量读取数据
        self.operation = self.client.dxl.GroupBulkRead(client.port_handler,
                                                       client.packet_handler)

        for motor_id in motor_ids:  # 遍历每个电机ID
            success = self.operation.addParam(motor_id, address, size)  # 添加参数到批量读取操作中
            if not success:  # 如果添加参数失败
                raise OSError(
                    '[Motor ID: {}] Could not add parameter to bulk read.'
                    .format(motor_id))  # 抛出异常

    def read(self, retries: int = 1):
        """Reads data from the motors."""
        self.client.check_connected()  # 检查客户端是否已连接
        success = False  # 初始化成功标志为False
        while not success and retries >= 0:  # 当未成功且重试次数大于等于0时
            comm_result = self.operation.txRxPacket()  # 发送和接收数据包
            success = self.client.handle_packet_result(
                comm_result, context='read')  # 处理数据包结果
            retries -= 1  # 减少重试次数

        # 如果读取失败，返回之前缓存的数据
        if not success:
            return self._get_data()

        errored_ids = []  # 初始化错误ID列表
        for i, motor_id in enumerate(self.motor_ids):  # 遍历每个电机ID
            # 检查数据是否可用
            available = self.operation.isAvailable(motor_id, self.address,
                                                   self.size)
            if not available:  # 如果数据不可用
                errored_ids.append(motor_id)  # 将电机ID添加到错误ID列表中
                continue

            self._update_data(i, motor_id)  # 更新缓存数据

        if errored_ids:  # 如果有错误ID
            logging.error('Bulk read data is unavailable for: %s',
                          str(errored_ids))  # 记录错误日志

        return self._get_data()  # 返回缓存的数据

    def _initialize_data(self):
        """Initializes the cached data."""
        self._data = np.zeros(len(self.motor_ids), dtype=np.float32)  # 初始化缓存数据为零数组

    def _update_data(self, index: int, motor_id: int):
        """Updates the data index for the given motor ID."""
        self._data[index] = self.operation.getData(motor_id, self.address,
                                                   self.size)  # 更新指定索引的缓存数据

    def _get_data(self):
        """Returns a copy of the data."""
        return self._data.copy()  # 返回缓存数据的副本


class DynamixelPosVelCurReader(DynamixelReader):
    """读取位置、速度和电流的类。"""

    def __init__(self,
                 client: DynamixelClient,
                 motor_ids: Sequence[int],
                 pos_scale: float = 1.0,
                 vel_scale: float = 1.0,
                 cur_scale: float = 1.0):
        """初始化读取器。

        Args:
            client: Dynamixel客户端实例。
            motor_ids: 电机ID列表。
            pos_scale: 位置缩放比例。
            vel_scale: 速度缩放比例。
            cur_scale: 电流缩放比例。
        """
        super().__init__(
            client,
            motor_ids,
            address=ADDR_PRESENT_POS_VEL_CUR,  # 读取位置、速度和电流的起始地址
            size=LEN_PRESENT_POS_VEL_CUR,  # 读取数据的字节大小
        )
        self.pos_scale = pos_scale  # 保存位置缩放比例
        self.vel_scale = vel_scale  # 保存速度缩放比例
        self.cur_scale = cur_scale  # 保存电流缩放比例

    def _initialize_data(self):
        """初始化缓存数据。"""
        self._pos_data = np.zeros(len(self.motor_ids), dtype=np.float32)  # 初始化位置数据为零数组
        self._vel_data = np.zeros(len(self.motor_ids), dtype=np.float32)  # 初始化速度数据为零数组
        self._cur_data = np.zeros(len(self.motor_ids), dtype=np.float32)  # 初始化电流数据为零数组

    def _update_data(self, index: int, motor_id: int):
        """更新指定电机ID的数据。

        Args:
            index: 电机ID在列表中的索引。
            motor_id: 电机ID。
        """
        cur = self.operation.getData(motor_id, ADDR_PRESENT_CURRENT,
                                     LEN_PRESENT_CURRENT)  # 获取当前电流数据
        vel = self.operation.getData(motor_id, ADDR_PRESENT_VELOCITY,
                                     LEN_PRESENT_VELOCITY)  # 获取当前速度数据
        pos = self.operation.getData(motor_id, ADDR_PRESENT_POSITION,
                                     LEN_PRESENT_POSITION)  # 获取当前位置数据
        cur = unsigned_to_signed(cur, size=2)  # 将电流数据转换为有符号整数
        vel = unsigned_to_signed(vel, size=4)  # 将速度数据转换为有符号整数
        pos = unsigned_to_signed(pos, size=4)  # 将位置数据转换为有符号整数
        self._pos_data[index] = float(pos) * self.pos_scale  # 更新位置数据并应用缩放比例
        self._vel_data[index] = float(vel) * self.vel_scale  # 更新速度数据并应用缩放比例
        self._cur_data[index] = float(cur) * self.cur_scale  # 更新电流数据并应用缩放比例

    def _get_data(self):
        """返回数据的副本。"""
        return (self._pos_data.copy(), self._vel_data.copy(),
                self._cur_data.copy())  # 返回位置、速度和电流数据的副本


class DynamixelPosReader(DynamixelReader):
    """读取位置的类。"""

    def __init__(self,
                 client: DynamixelClient,
                 motor_ids: Sequence[int],
                 pos_scale: float = 1.0,
                 vel_scale: float = 1.0,
                 cur_scale: float = 1.0):
        """初始化读取器。

        Args:
            client: Dynamixel客户端实例。
            motor_ids: 电机ID列表。
            pos_scale: 位置缩放比例。
            vel_scale: 速度缩放比例。
            cur_scale: 电流缩放比例。
        """
        super().__init__(
            client,
            motor_ids,
            address=ADDR_PRESENT_POS_VEL_CUR,  # 读取位置、速度和电流的起始地址
            size=LEN_PRESENT_POS_VEL_CUR,  # 读取数据的字节大小
        )
        self.pos_scale = pos_scale  # 保存位置缩放比例

    def _initialize_data(self):
        """初始化缓存数据。"""
        self._pos_data = np.zeros(len(self.motor_ids), dtype=np.float32)  # 初始化位置数据为零数组

    def _update_data(self, index: int, motor_id: int):
        """更新指定电机ID的数据。

        Args:
            index: 电机ID在列表中的索引。
            motor_id: 电机ID。
        """
        pos = self.operation.getData(motor_id, ADDR_PRESENT_POSITION,
                                     LEN_PRESENT_POSITION)  # 获取当前位置数据
        pos = unsigned_to_signed(pos, size=4)  # 将位置数据转换为有符号整数
        self._pos_data[index] = float(pos) * self.pos_scale  # 更新位置数据并应用缩放比例

    def _get_data(self):
        """返回数据的副本。"""
        return self._pos_data.copy()  # 返回位置数据的副本

class DynamixelVelReader(DynamixelReader):
    """读取速度的类。"""

    def __init__(self,
                 client: DynamixelClient,
                 motor_ids: Sequence[int],
                 pos_scale: float = 1.0,
                 vel_scale: float = 1.0,
                 cur_scale: float = 1.0):
        """初始化读取器。

        Args:
            client: Dynamixel客户端实例。
            motor_ids: 电机ID列表。
            pos_scale: 位置缩放比例。
            vel_scale: 速度缩放比例。
            cur_scale: 电流缩放比例。
        """
        super().__init__(
            client,
            motor_ids,
            address=ADDR_PRESENT_POS_VEL_CUR,  # 读取位置、速度和电流的起始地址
            size=LEN_PRESENT_POS_VEL_CUR,  # 读取数据的字节大小
        )
        self.pos_scale = pos_scale  # 保存位置缩放比例
        self.vel_scale = vel_scale  # 保存速度缩放比例
        self.cur_scale = cur_scale  # 保存电流缩放比例

    def _initialize_data(self):
        """初始化缓存数据。"""
        self._vel_data = np.zeros(len(self.motor_ids), dtype=np.float32)  # 初始化速度数据为零数组

    def _update_data(self, index: int, motor_id: int):
        """更新指定电机ID的数据。

        Args:
            index: 电机ID在列表中的索引。
            motor_id: 电机ID。
        """
        vel = self.operation.getData(motor_id, ADDR_PRESENT_VELOCITY,
                                     LEN_PRESENT_VELOCITY)  # 获取当前速度数据
        vel = unsigned_to_signed(vel, size=4)  # 将速度数据转换为有符号整数
        self._vel_data[index] = float(vel) * self.vel_scale  # 更新速度数据并应用缩放比例

    def _get_data(self):
        """返回数据的副本。"""
        return self._vel_data.copy()  # 返回速度数据的副本

class DynamixelCurReader(DynamixelReader):
    """读取电流的类。"""

    def __init__(self,
                 client: DynamixelClient,
                 motor_ids: Sequence[int],
                 pos_scale: float = 1.0,
                 vel_scale: float = 1.0,
                 cur_scale: float = 1.0):
        """初始化读取器。

        Args:
            client: Dynamixel客户端实例。
            motor_ids: 电机ID列表。
            pos_scale: 位置缩放比例。
            vel_scale: 速度缩放比例。
            cur_scale: 电流缩放比例。
        """
        super().__init__(
            client,
            motor_ids,
            address=ADDR_PRESENT_POS_VEL_CUR,  # 读取位置、速度和电流的起始地址
            size=LEN_PRESENT_POS_VEL_CUR,  # 读取数据的字节大小
        )
        self.cur_scale = cur_scale  # 保存电流缩放比例

    def _initialize_data(self):
        """初始化缓存数据。"""
        self._cur_data = np.zeros(len(self.motor_ids), dtype=np.float32)  # 初始化电流数据为零数组

    def _update_data(self, index: int, motor_id: int):
        """更新指定电机ID的数据。

        Args:
            index: 电机ID在列表中的索引。
            motor_id: 电机ID。
        """
        cur = self.operation.getData(motor_id, ADDR_PRESENT_CURRENT,
                                     LEN_PRESENT_CURRENT)  # 获取当前电流数据
        cur = unsigned_to_signed(cur, size=2)  # 将电流数据转换为有符号整数
        self._cur_data[index] = float(cur) * self.cur_scale  # 更新电流数据并应用缩放比例

    def _get_data(self):
        """返回数据的副本。"""
        return self._cur_data.copy()  # 返回电流数据的副本


# 注册全局清理函数。
atexit.register(dynamixel_cleanup_handler)

if __name__ == '__main__':
    import argparse  # 导入argparse模块，用于解析命令行参数
    import itertools  # 导入itertools模块，用于创建迭代器

    parser = argparse.ArgumentParser()  # 创建ArgumentParser对象
    parser.add_argument(
        '-m',
        '--motors',
        required=True,
        help='Comma-separated list of motor IDs.')  # 添加motors参数，用于指定电机ID列表
    parser.add_argument(
        '-d',
        '--device',
        default='/dev/ttyUSB0',
        help='The Dynamixel device to connect to.')  # 添加device参数，用于指定连接的设备
    parser.add_argument(
        '-b', '--baud', default=1000000, help='The baudrate to connect with.')  # 添加baud参数，用于指定波特率
    parsed_args = parser.parse_args()  # 解析命令行参数

    motors = [int(motor) for motor in parsed_args.motors.split(',')]  # 将电机ID列表转换为整数列表

    way_points = [np.zeros(len(motors)), np.full(len(motors), np.pi)]  # 创建两个路径点，一个全为零，一个全为π

    with DynamixelClient(motors, parsed_args.device,
                         parsed_args.baud) as dxl_client:  # 创建Dynamixel客户端并连接
        for step in itertools.count():  # 无限循环，step从0开始递增
            if step > 0 and step % 50 == 0:  # 每50步执行一次
                way_point = way_points[(step // 100) % len(way_points)]  # 选择路径点
                print('Writing: {}'.format(way_point.tolist()))  # 打印当前写入的路径点
                dxl_client.write_desired_pos(motors, way_point)  # 写入目标位置
            read_start = time.time()  # 记录读取开始时间
            pos_now, vel_now, cur_now = dxl_client.read_pos_vel_cur()  # 读取当前位置、速度和电流
            if step % 5 == 0:  # 每5步执行一次
                print('[{}] Frequency: {:.2f} Hz'.format(
                    step, 1.0 / (time.time() - read_start)))  # 打印当前频率
                print('> Pos: {}'.format(pos_now.tolist()))  # 打印当前位置
                print('> Vel: {}'.format(vel_now.tolist()))  # 打印当前速度
                print('> Cur: {}'.format(cur_now.tolist()))  # 打印当前电流
