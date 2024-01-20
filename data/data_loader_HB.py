import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
# from utils.tools import StandardScaler
from sklearn.preprocessing import StandardScaler
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

class DatabaseHandle:
    base_table_name = "main"

    def __init__(self, database_path):
        self.connection = sqlite3.connect(database_path)

    def select_distinct(self, column_name, table=None):
        if table is None:
            table = self.base_table_name
        print("分析数据库中...")
        rtn = pd.read_sql_query(f"select DISTINCT {column_name} from {table}",
                                self.connection)
        print("分析完成...")
        return rtn[column_name]

    def select_by(self, key, column_name, select="*", table=None):
        if table is None:
            table = self.base_table_name
        return pd.read_sql_query("select {} from {} where {}=='{}';".format(select, table, column_name, key),
                                 self.connection)

    def close(self):
        self.connection.close()


class FlightPathDatabaseHandle(DatabaseHandle):
    base_table_name = "fw_flightHJ"
    main_key_name = "HBID"
    data_buffer = {}
    def __init__(self, database_path):
        super().__init__(database_path)
        self.main_keys = self.select_distinct(self.main_key_name)

    def select_by(self, key, column_name=None, select="*", table=None):
        if column_name is None:
            column_name = self.main_key_name
        return super().select_by(key, column_name, select, table)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, item):
        if item in self.data_buffer.keys():
            return self.data_buffer[item]
        else:
            raw_data = self.select_by(self.main_keys[item])
            data = FlightPathDataFrame(raw_data)
            self.data_buffer.update({item:data})
            return data

    def __len__(self):
        return len(self.main_keys)


class FlightPathDataFrame(pd.DataFrame):
    """
    到这一步为止，数据原封不动，没有走任何变化处理
    """
    def __init__(self,df,*args,**kwargs):
        super().__init__(df,*args,**kwargs)
        self.to_datetime("WZSJ")#.to_datetime("RKSJ")

    def to_datetime(self,column):
        self[column] = pd.to_datetime(self[column])
        return self
    
def resample_dataframe(arr, n):
    df = pd.DataFrame(arr)
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], format='%Y-%m-%dT%H:%M:%S.%f')

    df = df.set_index(df.columns[0])
    df = df.iloc[:, 0:].apply(pd.to_numeric)
    
    
    df=df[~df.index.duplicated()]
    # df = df.interpolate(method='linear')
    # df = df.interpolate(method='linear')
    df_resampled = df.resample(n).bfill(limit=1).interpolate(method='linear')

    
    return df_resampled

class Dataset_flight(Dataset):
    def __init__(self, data,in_len,out_len,data_split = [0.7, 0.3],flag='train'):

        self.data_split = data_split
        assert flag in ['train', 'test']
        type_map = {'train':0,'test':1}
        self.set_type = type_map[flag]
        self.data = data
        self.in_len = in_len
        self.out_len = out_len
        self.__read_data__()

    def __read_data__(self):
        train_num = int(len(self.data)*self.data_split[0])
        test_num = int(len(self.data)*self.data_split[1])
        border1s = [0, train_num]
        border2s = [train_num, train_num + test_num]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        self.data_x = self.data[border1:border2,0:self.in_len,:]
        self.data_y = self.data[border1:border2,self.in_len:self.in_len+self.out_len,:]


    
    def __getitem__(self, index):
        seq_x = self.data_x[index]
        seq_y = self.data_y[index]
        return seq_x, seq_y
    
    def __len__(self):
        return len(self.data_x)

def sliding_window(matrix, window_len, n):
        new_shape = (1 + (matrix.shape[0] - window_len) // n, window_len, matrix.shape[1])
        new_matrix = np.zeros(new_shape)
        for i in range(new_shape[0]):
            new_matrix[i] = matrix[i * n : i * n + window_len]
        if (new_shape[0] - 1) * n + window_len < matrix.shape[0]:
            new_matrix = np.concatenate((new_matrix, matrix[-window_len:, :][np.newaxis, :, :]), axis=0)
        return new_matrix
    
def read_data(data_path,in_len,out_len,seg_d):
    flight_raw = FlightPathDatabaseHandle(data_path)


    win_len = in_len + out_len
    sig_data = []
    count=0

    for sample in flight_raw:
        count+=1
        a = sample.iloc[:, [1, 2, 3, 4, 5]].values

        a = a[~np.isnan(a[:,1:].astype(float)).any(axis=1)]
        # for idx in range(a.shape[0]):
        #     c= a[idx,1:].astype(float)
        #     if not(np.any(np.isnan(c))):
        #         a = a[idx:,:]
        #         break

                
        # plt.figure(figsize=(9, 6), dpi=150)
        # plt.scatter(a[:,1],a[:,2],color='r',s=10)
        # plt.show()
        if not np.any(a):
            continue
        b = resample_dataframe(a,'10S')
        b = b.iloc[:,:].values
        
#         acceleration = np.diff(b[:, 2:4],axis=0)  # 使用numpy的差分函数，计算速度列的差分，得到加速度
#         #acceleration = np.expand_dims(acceleration, axis=1)
#         filled_acceleration = np.insert(acceleration, 0, acceleration[0],axis=0)
#         # diff_h = np.diff(b[:, 2])
#         # filled_acceleration = np.insert(acceleration, 0, acceleration[0])
#         # filled_acceleration = np.insert(acceleration, 0, acceleration[0])
# # 创建新的5维轨迹数据
#         #complex_trace=np.expand_dims(b[...,0]+b[...,1]*1j, axis=1)
#         #spectrum = np.fft.fft(complex_trace)
#         b = np.concatenate((b, filled_acceleration),axis=-1)
#         #b = np.concatenate((b, filled_acceleration,spectrum.real,spectrum.imag),axis=-1)
        
        if np.any(np.isnan(b)):
            raise ValueError("NaN exist")
        # plt.figure(figsize=(9, 6), dpi=150)
        # plt.scatter(a[:,0],a[:,1],color='r',s=10)
        # plt.show()
        if b.shape[0] < win_len:
            continue
        sig_data.append(sliding_window(b,win_len,seg_d))
        if count == 10000:
            break
    
    percent = 1
    sig_data=sig_data[0:int(len(sig_data)*percent)]
    
    res_data = np.concatenate(sig_data,axis=0)
    for i in range(res_data.shape[-1]):
        res_data[:,:,i] = StandardScaler().fit_transform(res_data[:,:,i].T).T
    # res_data[:,:,1] = StandardScaler().fit_transform(res_data[:,:,1].T).T
    # res_data[:,:,2] = StandardScaler().fit_transform(res_data[:,:,2].T).T
    # res_data[:,:,3] = StandardScaler().fit_transform(res_data[:,:,3].T).T
    # res_data[:,:,4] = StandardScaler().fit_transform(res_data[:,:,4].T).T
    # ress_data = res_data[:,:,0:2]
    return res_data




        









# def sliding_window(matrix, window_len, n):
#     new_shape = (1 + (matrix.shape[0] - window_len) // n, window_len, matrix.shape[1])
#     new_matrix = np.zeros(new_shape)
#     for i in range(new_shape[0]):
#         new_matrix[i] = matrix[i * n : i * n + window_len]
#     if (new_shape[0] - 1) * n + window_len < matrix.shape[0]:
#         new_matrix = np.concatenate((new_matrix, matrix[-window_len:, :][np.newaxis, :, :]), axis=0)
#     return new_matrix



if __name__ == '__main__':
    database_path = 'FW.sqlite'
    data_set = Dataset_flight(database_path, 
                              10, 
                              size=[25, 40],
                              data_split = [0.7, 0.3],
                              flag='train'
    )
    data_loader = DataLoader(
        data_set,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )
    steps = len(data_loader)
    for i, (batch_x, batch_y) in enumerate(data_loader):
        a=1











    flight_path = FlightPathDatabaseHandle(database_path)

    # sig_data = []
    # count=0
    # for sample in flight_path:
    #     count+=1
    #     a = sample.iloc[:, [3, 4, 5, 6]].values
    #     sig_data.append(sliding_window(a,50,20))
    #     if count == 20:
    #         break
    # res_data = np.concatenate(sig_data,axis=0)
    # rr = res_data[2,0:20,:]
    # 可以直接索引
    for i in range(10):
        print(i, "--\n", flight_path[i])

    # 可以生成迭代对象
    count = 0
    for sample in flight_path:
        count+=1
        print(count, "--\n",sample)
        if count==20:
            break