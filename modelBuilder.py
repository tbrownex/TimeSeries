import tensorflow as tf
import os

class Model():
    def __init__(self):
        self.model = tf.keras.Sequential()
    def build_model(self, modelCfg):
        for layer in modelCfg['layers']:
            neurons = layer['neurons'] if 'neurons' in layer else None
            dropout_rate = layer['rate'] if 'rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_seq'] if 'return_seq' in layer else None
            input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
            input_dim = layer['input_dim'] if 'input_dim' in layer else None

            if layer['type'] == 'dense':
                self.model.add(tf.keras.layers.Dense(neurons, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(tf.keras.layers.LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq))
            if layer['type'] == 'dropout':
                self.model.add(tf.keras.layers.Dropout(dropout_rate))
            
        self.model.compile(loss=modelCfg['loss'], optimizer=modelCfg['optimizer'])
        print('[Model] Model Compiled')
    def train(self, x, y, epochs, batch_size, save_dir):
        print('[Model] Training Started')
        print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
        
        save_fname = save_dir+"tom"
        '''callbacks = [
                EarlyStopping(monitor='val_loss', patience=2),
                ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
        ]'''
        callbacks = None
        self.model.fit(
                x,
                y,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks
        )
        self.model.save(save_fname)

        print('[Model] Training Completed. Model saved as %s' % save_fname)