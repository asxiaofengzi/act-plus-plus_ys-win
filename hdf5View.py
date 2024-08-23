import h5py

def check_hdf5_file(file_path):
    try:
        with h5py.File(file_path, 'r') as f:
            # 列出所有对象（组和数据集）
            def printname(name):
                print(name)
            f.visit(printname)
    except Exception as e:
        print(f"Error opening file {file_path}: {e}")

# 替换为你的文件路径
file_path = 'data/sim_transfer_cube_scripted/episode_2.hdf5'
check_hdf5_file(file_path)
