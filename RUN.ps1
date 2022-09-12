for($i = 0; $i -lt 3; $i++){
    Write-Host "[$i] Script iteration" 
    python.exe .\src\01_MNIST_SVM.py
    python.exe .\src\02_MNIST_SVM_PCA.py
    python.exe .\src\03_MNIST_NN_SKLearn.py
    python.exe .\src\04_MNIST_NN_Keras.py
    Write-Host ""
    Write-Host ""
}
Write-Host "Script finished."
Write-Host ""