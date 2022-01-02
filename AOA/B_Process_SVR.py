from sklearn.svm import SVR
from B_Process_Generic import Process


class ProcessSVR(Process):
    def __init__(self, base_file_name, index, model_name):
        super().__init__(base_file_name, index, model_name)
        self.model = SVR(C=10.0, cache_size=200, degree=100, epsilon=0.001,
                         gamma=0.1, kernel='rbf',
                         max_iter=-1, shrinking=True, tol=0.001, verbose=False)


if __name__ == '__main__':
    process = ProcessSVR(base_file_name='E0a.txt_shuffle.csv', index=0, model_name='svr')
    process.process()
