import csv  # 导入csv模块，用于处理CSV文件
import os  # 导入os模块，用于文件和目录操作
import re  # 导入re模块，用于正则表达式操作

def find_largest_suffix(directory, base_filename, extension):
    """
    Finds the largest numeric suffix in filenames within a given directory,
    based on a specified base filename and extension.

    Parameters:
    directory (str): The path to the directory containing the files.
    base_filename (str): The base name of the files to search for.
    extension (str): The file extension of the files to search for.

    Returns:
    int: The largest numeric suffix found in the filenames. Returns -1 if no valid files are found.
    """
    max_suffix = -1  # 初始化最大后缀为-1，表示初始没有找到有效文件

    # 动态构建一个正则表达式模式，基于base_filename和extension
    pattern = re.compile(rf'{re.escape(base_filename)}_(\d+){re.escape(extension)}$')

    # 遍历给定目录中的所有文件
    for filename in os.listdir(directory):
        match = pattern.match(filename)  # 匹配文件名
        if match:
            # 提取匹配到的数字部分（第1组）并转换为整数
            num = int(match.group(1))
            # 如果这个数字比当前的max_suffix大，则更新max_suffix
            max_suffix = max(max_suffix, num)

    return max_suffix  # 返回最大的后缀

directory_path = 'logs'  # 定义目录路径
base_filename = 'testfitye'  # 定义基础文件名
extension = '.csv'  # 定义文件扩展名
largest_suffix = find_largest_suffix(directory_path, base_filename, extension)  # 查找最大的后缀
print(f"The largest suffix found in files starting with '{base_filename}' and ending with '{extension}' is: {largest_suffix}")  # 打印结果
