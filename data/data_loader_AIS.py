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

import os
import numpy as np
import pandas as pd

# 定义处理函数，用于解析文件并转换为numpy数组
def process_file(file_path):
    
    df = pd.read_csv(file_path, delimiter=' ', header=None)
    df[0] = df[0].str.replace('T', '')   
    # 解析每行数据并将其转换为numpy数组
    # 将DataFrame转换为numpy数组
    numpy_array = df.to_numpy(dtype=np.float64,na_value=0)
    numpy_array = numpy_array[:, np.concatenate([np.arange(1, numpy_array.shape[1]), [0]])]
    
    # df = pd.read_csv('./example.csv', sep=' ', header=None)

    # # 把最后一列时间字符串转换为日期格式
    # df[4] = pd.to_datetime(df[4] + ' ' + df[5]).astype(int) / 10**9
    # # 删除原始的时间列
    # df = df.drop(5, axis=1)
    # df = df.dropna()
    # df = pd.DataFrame(df)
    # numpy_array = df.values
    
    return numpy_array

def Readcsv(dataset_folder):
    """
    返回嵌套列表，第一层是类别，第二层是某一类的轨迹样本
    """
    # # 设置数据集文件夹路径
    # dataset_folder = '/mnt/d/haitundata'

    # 遍历标签文件夹
    label_folders = [label_folder for label_folder in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, label_folder))]
    label_folders.sort(key=lambda x:int(x.split('.')[0]))
    # 使用多进程处理每个标签文件夹下的文件

    result = []
    for label_folder in label_folders:
        label_folder_path = os.path.join(dataset_folder, label_folder)
        files = [os.path.join(label_folder_path, file) for file in os.listdir(label_folder_path) if os.path.isfile(os.path.join(label_folder_path, file))]
        result.append(list(map(process_file, files)))
    # result列表包含了所有文件的numpy数组

    return result

# if __name__ == "__main__":
#     print(len(Readcsv("/mnt/d/haitundata")[0]))

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
    df.iloc[:, -1] = pd.to_datetime(df.iloc[:, -1],unit='s')

    df = df.set_index(df.columns[-1])
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
        border1s = [0, train_num - self.in_len]
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
        return len(self.data_x) - self.in_len- self.out_len + 1

def sliding_window(matrix, window_len, n):
        new_shape = (1 + (matrix.shape[0] - window_len) // n, window_len, matrix.shape[1])
        new_matrix = np.zeros(new_shape)
        for i in range(new_shape[0]):
            new_matrix[i] = matrix[i * n : i * n + window_len]
        if (new_shape[0] - 1) * n + window_len < matrix.shape[0]:
            new_matrix = np.concatenate((new_matrix, matrix[-window_len:, :][np.newaxis, :, :]), axis=0)
        return new_matrix
    
def read_data(data_path,in_len,out_len,seg_d):
    AIS_raw = Readcsv(data_path)
    #AIS_list = [item for sublist in AIS_raw for item in sublist]
    AIS_list = [item for item  in AIS_raw[0]]
    win_len = in_len + out_len
    sig_data = []
    count=0

    for sample in AIS_list:
        count+=1
        # a = sample.iloc[:, [1, 2, 3, 4, 5]].values

        # a = a[~np.isnan(a[:,1:].astype(float)).any(axis=1)]
        # # for idx in range(a.shape[0]):
        # #     c= a[idx,1:].astype(float)
        # #     if not(np.any(np.isnan(c))):
        # #         a = a[idx:,:]
        # #         break

                
        # # plt.figure(figsize=(9, 6), dpi=150)
        # # plt.scatter(a[:,1],a[:,2],color='r',s=10)
        # # plt.show()
        # if not np.any(a):
        #     continue
        b = resample_dataframe(sample,'50S')
        b = b.iloc[:,:].values
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

    res_data = np.concatenate(sig_data,axis=0)
    res_data[:,:,0] = StandardScaler().fit_transform(res_data[:,:,0].T).T
    res_data[:,:,1] = StandardScaler().fit_transform(res_data[:,:,1].T).T
    res_data[:,:,2] = StandardScaler().fit_transform(res_data[:,:,2].T).T
    res_data[:,:,3] = StandardScaler().fit_transform(res_data[:,:,3].T).T
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