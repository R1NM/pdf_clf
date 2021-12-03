# Malicious Pdf Classifier w/ SVM
Requires python3 w/ scikit-learn & pandas packages <br>
Runs on Ubuntu

## Usage: 
### Training
*training data named "train.csv"
``` python
python3 model_training.py
```
### Prediction
*prediction data named "pre.csv"
``` python
python2 prediction.py
```

-input: pdf parsed w/ hidost<br>
-print prediction score
-result: list of malicious(0) / benign(1)
