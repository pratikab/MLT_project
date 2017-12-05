Webmail CAPTCHA

Classes and Classes_test stores segmented characters without noise removal
Classes1 and Classes_test1 stores segmented characters using boundary segmentation
Classes2 and Classes_test2 stores segmented characters using dominant colour segmentation
(refer report.pdf)

webmail_data contain 5000 webmail CAPTCHA
label1_4000.npy and label4001_5000.npy are labels of CAPTCHA present in webmail_data
model_len.py is the CNN used for training the number of characters (3 or 4) present in CAPTCHA
train_len_web.py is used to train the model for binary classiying the number of characters present in CAPTCHA using 4000 training dataset
train_len_web.pt saves the trained model for learning number of characters in CAPTCHA
test_len_web.py is used to test the trained model on 1000 CAPTCHA
test_len_single_web.py is used to test the trained model on 1 CAPTCHA

segmentation.py is used to segment the CAPTCHA after we know how many characters are present
model.py is the CNN used for training individual characters
train_web.py is used to train individual characters using CNN
trainmodel_web.pt stores the trained model
test_web.py is used for testing the trained model on individual characters
test_multiple.py is used to test whole CAPTCHA 
Accuracy on 1000 test dataset was 80.27%


