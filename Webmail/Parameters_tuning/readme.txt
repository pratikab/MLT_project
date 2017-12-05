Parameters tuning

model.py is the CNN model used for training and tuning parameters
Classes_train contains segmented characters of 3000 training CAPTCHA
Classes_validate contains segmented characters of 1000 validation CAPTCHA

train_batchsize_web.py is used to tune batchsize, it will train on Classes_train and will give accuracy on Classes_validate for batch size 5, 10, 15, .......45, 50

train_epochs_web.py is used to tune epochs (number of times same dataset is run), it will train on Classes_train and will give accuracy on Classes_validate for epochs 1, 2, 3, ..........19, 20
