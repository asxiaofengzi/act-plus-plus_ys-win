from interbotix_xs_modules.arm import InterbotixManipulatorXS  # 导入InterbotixManipulatorXS类，用于控制机械臂
from aloha_scripts.robot_utils import move_arms, torque_on, move_grippers  # 导入自定义的机械臂控制函数
from constants import PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE  # 导入夹爪打开和关闭的常量
import argparse  # 导入argparse模块，用于解析命令行参数
import numpy as np  # 导入NumPy库，用于数值计算

# 用于校准头部摄像头和机械臂对称性

def main():
    argparser = argparse.ArgumentParser()  # 创建ArgumentParser对象
    argparser.add_argument('--all', action='store_true', default=False)  # 添加命令行参数--all
    args = argparser.parse_args()  # 解析命令行参数

    # 初始化左侧机械臂对象
    puppet_bot_left = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_left', init_node=True)
    # 初始化右侧机械臂对象
    puppet_bot_right = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name=f'puppet_right', init_node=False)

    all_bots = [puppet_bot_left, puppet_bot_right]  # 将两个机械臂对象存储在列表中
    for bot in all_bots:
        torque_on(bot)  # 打开每个机械臂的扭矩
    
    multiplier = np.array([-1, 1, 1, -1, 1, 1])  # 定义一个乘数数组，用于生成对称位置
    puppet_sleep_position_left = np.array([-0.8, -0.5, 0.5, 0, 0.65, 0])  # 定义左侧机械臂的休眠位置
    puppet_sleep_position_right = puppet_sleep_position_left * multiplier  # 生成右侧机械臂的对称休眠位置
    all_positions = [puppet_sleep_position_left, puppet_sleep_position_right]  # 将两个位置存储在列表中
    move_arms(all_bots, all_positions, move_time=2)  # 移动机械臂到指定位置，移动时间为2秒

    # move_grippers(all_bots, [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=1)  # 打开夹爪，移动时间为1秒（此行被注释掉了）

if __name__ == '__main__':
    main()  # 如果此脚本是主程序，则执行main函数
