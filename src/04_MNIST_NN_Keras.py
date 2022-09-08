# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# %matplotlib inline
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
import seaborn as sns


### Configurations
# Training-Size
num_train = 15000                   # 60000 for full data set 
num_test  = 2500                    # 10000 for full data set


# Use GridSearchCV to look up optimal parameters (see below)
hyper_parameter_search = True       # True/False: Run hyper-parameter search via GridSearchCV. Takes a long time.

# Simple function to log information
txt_out_file_path = 'keras-nn-hyperparameter-tuning-log.txt'
def print_to_txt_file(*s):
    with open(txt_out_file_path, 'a') as f:
        for arg in s:
            print(arg, file=f)
            print(arg)

# Fetch MNIST-Data from Keras repository
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Display (Train) (Test) datasets
print("Data : Dataset Trainingset")
print(X_train.shape, X_test.shape)
print("Labels : Dataset Trainingset")
print(y_train.shape, y_test.shape)

# i.e.: We have 60000 images with a size of 28x28 pixels

# Visualize some examples
num_classes = 10 # 0 .. 9
f, ax = plt.subplots(1, num_classes, figsize=(20,20))
for i in range(0, num_classes):
  sample = X_train[y_train == i][0]
  ax[i].imshow(sample, cmap='gray')
  ax[i].set_title("Label: {}".format(i), fontsize=16)
  ax[i].axis('off')

# Reshape the data such that we have access to every pixel of the image
# The reason to access every pixel is that only then we can apply deep learning ideas and can assign color code to every pixel.
train_data = X_train.reshape((X_train.shape[0], 28*28)).astype('float32')
train_label = y_train.astype("float32")

test_data = X_test.reshape((X_test.shape[0], 28*28)).astype('float32')
test_label = y_test.astype("float32")

# We know the RGB color code where different values produce various colors. It is also difficult to remember every color combination. 
# We already know that each pixel has its unique color code and also we know that it has a maximum value of 255. 
# To perform Machine Learning, it is important to convert all the values from 0 to 255 for every pixel to a range of values from 0 to 1.
train_data = train_data / 255
test_data = test_data / 255

# Force the amount of columns to fit the necessary sizes required by the neural network
train_label = keras.utils.to_categorical(train_label, num_classes)
test_label = keras.utils.to_categorical(test_label, num_classes)

# As an optional step, we decrease the training and testing data size, such that the algorithms perform their execution in acceptable time
train_data = train_data[1:num_train,]
train_label = train_label[1:num_train]

test_data = test_data[1:num_test,]
test_label = test_label[1:num_test]


# Display (Train) (Test) datasets
print("Reshaped Data : Dataset Trainingset")
print(train_data.shape, test_data.shape)
print("Reshaped Labels : Dataset Trainingset")
print(train_label.shape, test_label.shape)

# As we can see: We now have X images with 784 pixels in total
# We now operate on this data


# Create model: https://keras.io/guides/sequential_model/
model = Sequential()

# Create model layers
model.add(Dense(units=128, input_shape=(784,), activation='relu'))
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Additional learning configurations
batch_size = 512
epochs=10

# Train model
model.fit(x=train_data, y=train_label, batch_size=batch_size, epochs=epochs)

# Evaluate model based on supplied tags
test_loss, test_acc = model.evaluate(test_data, test_label)
print_to_txt_file("--- [%s] Running Keras-MLP Classifier ---" % datetime.now())
print_to_txt_file("Test Loss: %s" % test_loss)
print_to_txt_file("Test Accuracy: %s" % test_acc)

# Let model predict data
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)

print(y_pred)
print(y_pred_classes)

# View a correctly predicted datapoint
random_idx = np.random.choice(len(test_data))
x_sample = test_data[random_idx]
y_true = np.argmax(test_label, axis=1)
y_sample_true = y_true[random_idx]
y_sample_pred_class = y_pred_classes[random_idx]

plt.title("Predicted: {}, True: {}".format(y_sample_pred_class, y_sample_true), fontsize=16)
plt.imshow(x_sample.reshape(28, 28), cmap='gray')

# Visualize estimation over correct and incorrect prediction via confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)

# Plot
fig, ax = plt.subplots(figsize=(15,10))
ax = sns.heatmap(confusion_mtx, annot=True, fmt='d', ax=ax, cmap="Blues")
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix');

# Review some Errors
# Create some sets of data
errors = (y_pred_classes - y_true != 0)
y_pred_classes_errors = y_pred_classes[errors]
y_pred_errors = y_pred[errors]
y_true_errors = y_true[errors]
x_test_errors = test_data[errors]

# Aggregate error set results
y_pred_errors_probability = np.max(y_pred_errors, axis=1)
true_probability_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))
diff_errors_pred_true = y_pred_errors_probability - true_probability_errors

# Get list of indices of sorted differences
sorted_idx_diff_errors = np.argsort(diff_errors_pred_true)
top_idx_diff_errors = sorted_idx_diff_errors[-5:] # 5 last ones

# Show error points which were the most failing ones
num = len(top_idx_diff_errors)
f, ax = plt.subplots(1, num, figsize=(30,30))

for i in range(0, num):
  idx = top_idx_diff_errors[i]
  sample = x_test_errors[idx].reshape(28,28)
  y_t = y_true_errors[idx]
  y_p = y_pred_classes_errors[idx]
  ax[i].imshow(sample, cmap='gray')
  ax[i].set_title("Predicted label :{}\nTrue label: {}".format(y_p, y_t), fontsize=22)
  ax[i].axis('off')