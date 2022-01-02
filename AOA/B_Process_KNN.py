from sklearn.neighbors import KNeighborsRegressor
from B_Process_Generic import Process


class ProcessKNN(Process):
    def __init__(self, base_file_name, index):
        super().__init__(base_file_name, index)
        self.model = KNeighborsRegressor(n_neighbors=20)


if __name__ == '__main__':
    process = ProcessKNN(base_file_name='E0a.txt_shuffle.csv', index=0)
    process.process()
