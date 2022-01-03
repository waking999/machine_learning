from sklearn.linear_model import LinearRegression
from AOA.generic.B_Process_Generic import Process


class ProcessLP(Process):
    def __init__(self, base_file_name, dataset_index, model_name):
        super().__init__(base_file_name, dataset_index, model_name)
        self.model = LinearRegression()


if __name__ == '__main__':
    process = ProcessLP(base_file_name='E0a.txt_shuffle.csv', dataset_index=0, model_name='lp')
    process.process()
