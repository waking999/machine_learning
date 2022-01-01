import matplotlib.pyplot as plt


class Plot:
    def __init__(self):
        return

    @staticmethod
    def plot(train_x, train_y, val_x, val_y, pred_y):
        plt.scatter(train_x, train_y, marker='o', c='black')
        plt.scatter(val_x, val_y, marker='o', c='black')

        plt.scatter(val_x, pred_y, marker='*', c='red')

        plt.show()
