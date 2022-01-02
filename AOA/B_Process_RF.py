from sklearn.ensemble import RandomForestRegressor
from B_Process_Generic import Process


class ProcessRF(Process):
    def __init__(self, base_file_name, index, model_name):
        super().__init__(base_file_name, index, model_name)
        self.model = RandomForestRegressor(max_depth=2, random_state=0, n_estimators=100)


if __name__ == '__main__':
    process = ProcessRF(base_file_name='E0a.txt_shuffle.csv', index=0, model_name='rf')
    process.process()
