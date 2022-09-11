$start=0
$end=2
for ($i = $start; $i < $end; i++){
    echo "[$i/$end] Script iteration" 
    python.exe .\src\01_MNIST_SVM.py
    python.exe .\src\02_MNIST_SVM_PCA.py
    python.exe .\src\03_MNIST_NN_SKLearn.py
    python.exe .\src\04_MNIST_NN_Keras.py
    echo ""
    echo ""
}
echo "Script finished."
echo ""