from keras.layers import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.models import Model
import tensorflow as tf
from keras.utils import plot_model
import numpy as np


# %%
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


# %%
class CTC():

    def __init__(self,
                 input_size=None,
                 output_size=None):
        self.input_size = input_size
        self.output_size = output_size
        self.m = None
        self.tm = None

    def build(self,
              act='relu',
              LSTM_units=200,
              drop_out=0.8):

        i = Input(shape=self.input_size, name='input')

        for _ in range(1):
            x = (Conv1D(filters=28, kernel_size=14, padding='same'))(i)
            x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
            x = Activation(act)(x)

        for _ in range(1):
            x = Bidirectional(LSTM(LSTM_units, return_sequences=True))(x)
            x = Dropout(drop_out)(x)
            x = BatchNormalization(momentum=0.9, epsilon=1e-8)(x)

        y_pred = TimeDistributed(Dense(self.output_size,
                                       activation='softmax'))(x)

        # ctc inputs
        labels = Input(name='the_labels', shape=[None, ], dtype='int32')
        input_length = Input(name='input_length', shape=[1], dtype='int32')
        label_length = Input(name='label_length', shape=[1], dtype='int32')

        # Keras doesn't currently support loss funcs with extra parameters
        # so CTC loss is implemented in a lambda layer
        loss_out = Lambda(ctc_lambda_func,
                          output_shape=(1,),
                          name='ctc')([y_pred,
                                       labels,
                                       input_length,
                                       label_length])
        self.tm = Model(inputs=i,
                        outputs=y_pred)
        self.m = Model(inputs=[i,
                               labels,
                               input_length,
                               label_length],
                       outputs=loss_out)
        # plot_model(self.tm, to_file='1.png')
        # plot_model(self.m, to_file='2.png')
        return self.m, self.tm


batch = 50


# dummy loss
def ctc(y_true, y_pred):
    return y_pred


def get_ctc_params(data, label):
    train_in_len = []
    for i in range(len(data)):
        train_in_len.append(data[i].shape[0])

    train_label_len = []
    for i in range(len(label)):
        train_label_len.append(label[i].shape[0])

    train_in_len = np.array(train_in_len)
    train_label_len = np.array(train_label_len)
    return train_in_len, train_label_len


X_train = np.load('data/xtrain10.npy')
Y_train = np.load('data/ytrain10.npy')

X_val = np.load('data/xtest10.npy')
Y_val = np.load('data/ytest10.npy')

input_length, label_length = get_ctc_params(X_train, Y_train)
input_length_val, label_length_val = get_ctc_params(X_val, Y_val)

labels = Y_train
labels_val = Y_val

sr_ctc = CTC((None, 78), 4)
sr_ctc.build()

adam = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-5, clipnorm=1.0)

sr_ctc.m.compile(loss=ctc, optimizer=adam, metrics=['accuracy'])
sr_ctc.tm.compile(loss=ctc, optimizer=adam)

checkpointer = ModelCheckpoint(filepath="ctc_{}_best.h5".format(date),
                               verbose=0,
                               save_best_only=True)

history = sr_ctc.m.fit([np.squeeze(X_train),
                        labels,
                        input_length,
                        label_length],
                       np.zeros([np.array(len(X_train))]),
                       batch_size=batch,
                       epochs=19,
                       validation_data=([np.squeeze(X_val),
                                         labels_val,
                                         input_length_val,
                                         label_length_val],
                                        np.zeros([np.array(len(X_val))])),
                       callbacks=[checkpointer],
                       verbose=1,
                       shuffle=True)

sr_ctc.m.save_weights('ctc_{}.h5'.format(date))
sr_ctc.tm.load_weights('ctc_{}_best.h5'.format(date))


def str_out(dataset=X_val):
    k_ctc_out = K.ctc_decode(sr_ctc.tm.predict(np.squeeze(dataset),
                                               verbose=1),
                             np.array(input_length_val))
    decoded_out = K.eval(k_ctc_out[0][0])
    return decoded_out


y_pred_val = str_out()
print('Output:', y_pred_val.shape)
print(len(y_pred_val))
print(len(Y_val))

r = []
t = []
for i in range(0, len(Y_val)):
    r.append([j for j in y_pred_val[i] if j != -1 and j != 0])
    t.append([j for j in Y_val[i] if j != 3 and j != 0])

for i in range(0, len(Y_val)):
    print('Predicted:', r[i])
    print('Original:', t[i])
    print('\n')