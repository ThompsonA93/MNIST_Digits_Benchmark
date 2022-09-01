###############
# Init system #
############### 

# Load packages
library("e1071")
library("keras")


# Disable warnings
options(warn=-1)


# Clear workspace
graphics.off()
rm(list=ls())


# Experimental setup
num_train = 5000   # 60000 for full data set 
num_test  =  500   # 10000 for full data set



#########################
# Load and prepare data #
#########################

# Load MNIST data
mnist = dataset_mnist()


# Assign train and test data+labels
train_data  = mnist$train$x
train_label = mnist$train$y

test_data  = mnist$test$x
test_label = mnist$test$y


# Reshape images to vectors
train_data = array_reshape(train_data, c(nrow(train_data), 784))
test_data  = array_reshape(test_data, c(nrow(test_data), 784))


# Rescale data to range [0,1]
train_data = train_data / 255
test_data = test_data / 255


# select subset for training
train_data = train_data[1:num_train,]
train_label = train_label[1:num_train]


# select subset for testing
test_data = test_data[1:num_test,]
test_label = test_label[1:num_test]



################################
# Train and run classification #
################################

# Init timer
t1 = proc.time()


# Train SVM
S = svm(train_data, factor(train_label))


cat("\n\nCorrect classification results:")

# Eval SVM on training data
pr_tr = predict(S, train_data)
success = sum(pr_tr==factor(train_label))/length(train_label)*100
res_s = sprintf('\n   Train: %5.2f\n\n', success)
cat(res_s)


# Eval SVM on test data
pr_te = predict(S, test_data)
success = sum(pr_te==factor(test_label))/length(test_label)*100
res_s = sprintf('\n\n    Test: %5.2f\n\n', success)
cat(res_s)


# End time, calculate elapsed time
t2 = proc.time()
t = t2-t1
cat("Computation time:\n\n")
print(t)
