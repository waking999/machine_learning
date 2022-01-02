from sklearn.ensemble import GradientBoostingRegressor
from B_Process_Generic import Process


class ProcessGBDT(Process):
    def __init__(self, base_file_name, index, model_name):
        super().__init__(base_file_name, index, model_name)
        self.model = GradientBoostingRegressor(n_estimators=400, max_depth=11, learning_rate=0.06, loss='squared_error',
                                               subsample=0.8)


if __name__ == '__main__':
    process = ProcessGBDT(base_file_name='E0a.txt_shuffle.csv', index=0, model_name='gbdt')
    process.process()
