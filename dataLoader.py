import pandas as pd
import numpy as np

class getData():
    def __init__(self, fileName, testPct, cols):
        df = pd.read_csv(fileName)
        trainSize = int(len(df) * (1-testPct))
        self.train = df.get(cols).values[:trainSize]
        self.test  = df.get(cols).values[trainSize:]
        self.trainSize  = trainSize
        '''self.len_test   = len(self.data_test)
        self.len_train_windows = None'''

    def get_train_data(self, seq_len, normalise):
        data_x = []
        data_y = []
        for i in range(self.trainSize - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)
    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.train[i:i+seq_len]
        if normalise:
            window = self.normalise_windows(window, single_window=True)[0]
        x = window[:-1]
        y = window[-1, [0]]
        return x, y
    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
