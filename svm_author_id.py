#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import sklearn.svm as svm
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()
#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100] 
#features_test = features_test[:len(features_test)/100] 
#labels_test = labels_test[:len(labels_test)/100] 




#########################################################
### your code goes here ###
clf=svm.SVC(kernel='rbf',C=10000)
t0=time()
clf.fit(features_train,labels_train)
print "My fit time : {} seconds".format(round(time()-t0,2))
print "######################"
print "/n"
print "######################"
t1=time()
test=clf.predict(features_test)
print "prediction time: {} seconds".format(round(time()-t1,2)) 
print "######################"
print "/n"
print "######################"
 
origin=labels_test
Sara_test=0
Chris_test=0
Sara_origin=0
Chris_origin=0

for x in labels_test:
    if x==0:
        Sara_origin+=1
    elif x==1:
        Chris_origin+=1
print "Sara's default mail  : {}".format(Sara_origin)
print "Chris default mail  : {}".format(Chris_origin)      # calculate sum of labels_test

for x in test:
    if x==0:
        Sara_test+=1
    else:
        Chris_test+=1
        
print "Sara's mail predictions : {}".format(Sara_test)
print "Chris mail predictions: {}".format(Chris_test)       # calculate sum of predictions without envolving default labels_test
        
missfit=0
if Sara_test>Sara_origin:
    missfit+= Sara_test-Sara_origin

if Chris_test>Chris_origin:
    missfit+= Chris_test-Chris_origin
# if default labels(sum of them)< predictions something is wrong!!!
#e.g. some mails of sara are fitted as chris mails     
    
print "No. of missmatched features : {}".format(missfit)


print "Total number of labels : {}".format(len(labels_test))
correct_classified = len(labels_test)-missfit
accuracy=float(correct_classified)/float(len(labels_test))
print "The accuracy of the classifier is : {} % of {}".format(round(accuracy*100,1), accuracy)

print "######################"
print "/n"
print "######################"
print "10th element:{}".format(labels_test[10])
print "prediction of 10th element:{}".format(test[10])

print "######################"
print "/n"
print "######################"
print "26th element:{}".format(labels_test[26])
print "prediction of 26th element:{}".format(test[26])

print "######################"
print "/n"
print "######################"
print "50th element:{}".format(labels_test[50])
print "prediction of 50th element:{}".format(test[50])

print "######################"
print "/n"
print "######################"
print "100th element:{}".format(labels_test[100])
print "prediction of 100th element:{}".format(test[100])

#########################################################


