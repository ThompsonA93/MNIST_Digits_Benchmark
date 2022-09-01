
import time

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import CSVLogger


# Write to file
def logModel(s):
    with open('keras-model-log.txt', 'a') as f:
        print(s, file=f)

    
def run_keras_neurnet(x_train, y_train, x_test, y_test):
    print("+-+-+-+-+ Executing Keras Neural Net implementation +-+-+-+-+")

    input = keras.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(input)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    output = layers.Dense(10, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs=input, outputs=output)

    optimizer = keras.optimizers.RMSprop()
    loss = keras.losses.SparseCategoricalCrossentropy()
    metrics = [keras.metrics.SparseCategoricalAccuracy()]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    model.summary(print_fn=logModel)

    batch_size=64
    epochs=10
    csv_logger = CSVLogger('keras-training-log.csv', append=True, separator=',')
    time_start = time.time()
    learning_history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs,callbacks=[csv_logger])
    time_end = time.time()

    print("Evaluation of test data: ")
    results = model.evaluate(x_test, y_test, batch_size)
    print("\t[Loss, Accuracy] :", results)
    print("\t Total Time required: ", time_end - time_start, "s")

