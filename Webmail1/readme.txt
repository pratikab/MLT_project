# Breaking CSE_Squirrelmail CAPTCHA  
segmentation_web1.py is used to segment a given Webmail1 CAPTCHA into 6 individul characters  
Classes & Classes_test contain individual characters after segmentation for training and testing respectively  
model.py is the structure of CNN that will be trained  
train_web1.py is used to train the model
trainmodel_web1.pt stores the learned model which will be used in testing
test_web1.py is used to test the trained model on test dataset
test_single_web1.py will take input whole CAPTCHA image, segment it and then give output


