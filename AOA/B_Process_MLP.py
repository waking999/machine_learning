from sklearn.neural_network import MLPRegressor
from AOA.generic.B_Process_Generic import Process


class ProcessMLP(Process):
    def __init__(self, base_file_name, dataset_index, model_name):
        super().__init__(base_file_name, dataset_index, model_name)
        self.model = MLPRegressor(activation='relu', alpha=0.0001, batch_size='auto',
                                  learning_rate='constant',
                                  learning_rate_init=0.0001, power_t=0.5,
                                  max_iter=200000, momentum=0.9, shuffle=True, random_state=None,
                                  nesterovs_momentum=True, solver='adam', tol=0.0001,
                                  n_iter_no_change=10, validation_fraction=0.1, verbose=False,
                                  warm_start=True, hidden_layer_sizes=(
                512, 256, 128, 64, 32)
                                  )


if __name__ == '__main__':
    process = ProcessMLP(base_file_name='E0a.txt_shuffle.csv', dataset_index=0, model_name='mlp')
    process.process()
