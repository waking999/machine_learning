from sklearn.tree import DecisionTreeRegressor
from AOA.generic.B_Process_Generic import Process


class ProcessDT(Process):
    def __init__(self, base_file_name, dataset_index, model_name):
        super().__init__(base_file_name, dataset_index, model_name)
        self.model = DecisionTreeRegressor(max_depth=10)


if __name__ == '__main__':
    process = ProcessDT(base_file_name='E0a.txt_shuffle.csv', dataset_index=0, model_name='dt')
    process.process()
