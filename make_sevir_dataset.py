import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import h5py
import warnings
from collections import OrderedDict
from tqdm import trange

warnings.filterwarnings('ignore')


def remove_duplicates_and_keep_order(my_list):
    return list(OrderedDict.fromkeys(my_list))


class Aggregated_categories(Dataset):
    def __init__(self, data_path, csv_path, txt_path):
        """
        初始化数据集
        :param data_path: h5 数据文件所在的目录
        :param csv_path: CATALOG.csv 文件的路径
        :param txt_path: 记录 sample_id 的 txt 文件路径 (如 train.txt, test.txt)
        """
        self.data_path = data_path
        self.df = pd.read_csv(csv_path)

        with open(txt_path, 'r') as f:
            self.sample_id = f.readlines()

    def __len__(self):
        return len(self.sample_id)

    def __getitem__(self, idx):
        data_id = self.sample_id[idx].strip()

        # 解析文件名和索引
        file = data_id.split('.h5')[0] + '.h5'
        index1 = int(data_id.split('.h5')[1].split('-')[1])
        index2 = int(data_id.split('.h5')[1].split('-')[2].strip('.npy'))
        file = file.replace('-', '/')

        # 读取 h5 文件
        with h5py.File(os.path.join(self.data_path, file), 'r') as f:
            seq = f['vil'][index1]
            seq = (seq != 255) * seq
            time_id = f['id'][index1].decode('utf-8')

        # 切片逻辑
        if index2 == 0:
            seq = seq[:, :, 0:25]
        elif index2 == 1:
            seq = seq[:, :, 12:37]
        elif index2 == 2:
            seq = seq[:, :, 24:49]

        time_utc = self.df.loc[self.df['id'] == time_id]['time_utc'].values[0]
        seq = torch.from_numpy(np.array(seq)).float().permute(2, 0, 1)

        return {
            'sequence': seq,
            'id': time_utc,
            'index': index2,
            'data_id': data_id
        }


def main(mode):
    # 切换到脚本所在目录
    os.chdir(os.path.abspath(os.path.dirname(__file__)))

    # ================= 路径配置区 =================
    # 1. SEVIR 原始数据的根目录 (在这里修改你的路径)
    base_sevir_path = './sevir_dataset'

    # 2. 具体文件的路径 (基于基础路径拼接，也可单独修改)
    data_path = os.path.join(base_sevir_path, 'data')
    csv_path = os.path.join(base_sevir_path, 'CATALOG.csv')
    txt_path = os.path.join(base_sevir_path, f'{mode}.txt')

    # 3. 数据处理后的保存目录
    save_dir = os.path.join(base_sevir_path, f'cascast/{mode}')
    # ===============================================

    os.makedirs(save_dir, exist_ok=True)

    # 实例化数据集
    data = Aggregated_categories(data_path=data_path, csv_path=csv_path, txt_path=txt_path)

    # 遍历并保存数据 (使用 tqdm 显示进度条)
    for i in trange(len(data), desc=f"Processing {mode} data"):
        batch = data[i]
        name = batch['data_id']
        sequence = batch['sequence'].numpy()
        time_id = batch['id']
        index = batch['index']

        # 替换后缀并保存
        saved_path = os.path.join(save_dir, name).replace('.npy', '.npz')
        np.savez(saved_path, sequence=sequence, id=time_id, index=index)


if __name__ == '__main__':
    # 生成测试集
    main('test')