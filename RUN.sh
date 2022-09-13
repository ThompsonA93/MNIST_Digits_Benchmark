#!/bin/bash
start=0
end=0

for (( i=$start; i<=$end; i++ ))
do   
    echo "[$i/$end] Script iteration"
    python3 src/01_MNIST_SVM.py
    python3 src/02_MNIST_SVM_PCA.py
    python3 src/03_MNIST_NN_SKLearn.py
    python3 src/04_MNIST_NN_Keras.py
    echo ""
    echo ""
done
echo "Script finished."
echo ""
