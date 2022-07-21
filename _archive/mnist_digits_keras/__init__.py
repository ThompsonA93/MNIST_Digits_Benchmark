import matplotlib.pyplot as plt
from datetime import datetime

import time
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.callbacks import CSVLogger

training_size = 1000    
num_classes = 10        # 0 .. 9 

def logModel(s):
    with open('keras-model-log.txt', 'a') as f:
        print(s, file=f)


def start_keras_app(x_train, y_train, x_test, y_test):
    f, ax = plt.subplots(1, num_classes, figsize=(20,20))
    for i in range(0, num_classes):
        sample = x_train[y_train == i][0]
        ax[i].imshow(sample, cmap='gray')
        ax[i].set_title("Label: {}".format(i), fontsize=16)
        ax[i].axis('off')
    plt.show()


#####################################
# Model-def
#####################################
    input = keras.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(input)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    output = layers.Dense(10, activation='softmax', name='predictions')(x)
    model = keras.Model(inputs=input, outputs=output)

#####################################
# Optimizer
#####################################

    # SGD, RMSprop, ADAM, ...
    optimizer = keras.optimizers.RMSprop()

#####################################
# Loss
#####################################


    # MeanSquaredError, KLDivergence, CosineSimilarty
    #loss=setLossFunction(y_test, y_train, 1)
    loss=keras.losses.SparseCategoricalCrossentropy()
#####################################
# Metrics
#####################################
    # AUC, Precision, Recall
    metrics=[keras.metrics.SparseCategoricalAccuracy()]



    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()
    # Write to file


    model.summary(print_fn=logModel)

    batch_size=64
    epochs=10
    csv_logger = CSVLogger('keras-training-log.csv', append=True, separator=',')
    time_start = time.time()
    learning_history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),callbacks=[csv_logger])
    time_end = time.time()
    print("Evaluation of test data: ")
    results = model.evaluate(x_test, y_test, batch_size)
    print("\t[Loss, Accuracy] :", results)
    print("\t Total Time required: ", time_end - time_start, "s")