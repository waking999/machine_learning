from sklearn.neighbors import KNeighborsRegressor
from AOA.generic.B_Process_Generic import Process


class ProcessKNN(Process):
    def __init__(self, base_file_name, dataset_index, model_name):
        super().__init__(base_file_name, dataset_index, model_name)
        self.model = KNeighborsRegressor(n_neighbors=2)


if __name__ == '__main__':
    process = ProcessKNN(base_file_name='E0a.txt_shuffle.csv', dataset_index=0, model_name='knn')
    process.process()
