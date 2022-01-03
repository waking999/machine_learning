import xgboost as xgb
from AOA.generic.B_Process_Generic import Process


class ProcessXGB(Process):
    def __init__(self, base_file_name, index, model_name):
        super().__init__(base_file_name, index, model_name)
        self.model = xgb.XGBRegressor(max_depth=127, learning_rate=0.01, n_estimators=1000,
                                      objective='reg:tweedie', n_jobs=-1, booster='gbtree')


if __name__ == '__main__':
    process = ProcessXGB(base_file_name='E0a.txt_shuffle.csv', index=0, model_name='xgb')
    process.process()
