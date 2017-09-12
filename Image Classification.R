#Name: Yu-Chu Alice Chen 
#Class: APANK 4335 
#Homework 4 R code

#I will repeat the process of changing directory, file names, etc., with all 5 image 
##categories with the following 4 models.

#load libraries 
library(jpeg)
library(EBImage)
library(neuralnet)
library(MASS)
library(caret)
library("e1071")

#function for turning it into vector (+ resize)
myFUN <- function (name){
  name <- readImage(name)
  name<- resize(name, w = 30, h=30)
  temp<- as.array(name)
  temp2<- data.frame(temp)
  out<- unlist(temp2)
  return(out)
}

#load images 
setwd("~/Desktop/homework data/Leopards/Train")
trains<-list.files(getwd(), pattern = '.*jpg', full.names = TRUE)
tests<- list.files(getwd(), pattern = '.*jpg', full.names = TRUE)

##resize to same w&h - list, unlist and matrix
myfiles<-lapply(trains, myFUN)
train_data<-unlist(myfiles)
train_output<-matrix(train_data,nrow=138)

myfiles2<-lapply(tests, myFUN)
test_data<-unlist(myfiles2)
test_output<-matrix(test_data,nrow=62)

#next step cbind excel file 
setwd("~/Desktop/new data")
train_label<-read.csv("leopards_train_label.csv",header=TRUE)
class(train_label)
combine_train<-cbind(train_label, train_output)
combine_train$final = as.factor (combine_train$final) #except for neural network 
combine_train<-combine_train[,-1]

setwd("~/Desktop/new data")
test_label<-read.csv("leopards_test_label.csv",header=TRUE)
class(test_label)
combine_test<-cbind(test_label, test_output)
combine_test$final = as.factor (combine_test$final) #except for neural network 
combine_test<-combine_test[,-1]

#Logistic Regression
logist <- glm(formula = final~.,data=combine_train,family=binomial(link='logit'),maxit=80)
summary(logist)

##predict logistic model for train dataset
model_prob <-ifelse(predict(logist,combine_train,type="response")>0.5,1,0)
table<-table(`Actual class`=combine_train$final,`Predicted class`=model_prob)
confusionMatrix(table)

##predict logistic model for test dataset
model_probtest <-ifelse(predict(logist,combine_test,type="response")>0.5,1,0)
tabletest<-table(`Actual class`=combine_test$final,`Predicted class`=model_probtest)
confusionMatrix(tabletest)

#Neural Network
apply(combine_train, 2, range)
names(combine_train)[2:2701]<-paste("Var",2:2701,sep="")
allVars<-colnames(combine_train)
predictorVars<-allVars[!allVars%in%"final"]
predictorVars<-paste(predictorVars,collapse = "+")
form=as.formula(paste("final~",predictorVars,collapse = "+"))

##predict neural network for train set 
neuralModel<-neuralnet(formula = form, hidden = c(10,2),linear.output = TRUE,data=combine_train)
predictions<-compute(neuralModel,combine_train[,2:2701])

print(head(predictions$net.result))
predictions$net.result<-sapply(predictions$net.result, round, digits=0)
tab<-table(combine_train$final,predictions$net.result)
confusionMatrix(tab)

#predict neural network for test set 
predictions_test<-compute(neuralModel,combine_test[,2:2701])

print(head(predictions_test$net.result))
predictions_test$net.result<-sapply(predictions_test$net.result, round, digits=0)
tab_test<-table(combine_test$final,predictions_test$net.result)
confusionMatrix(tab_test)

#SVM
svm_model <- svm(final ~ ., data=combine_train)
summary(svm_model)

##predict SVM on training set
predsvm <- predict(svm_model,newdata=combine_train)
summary(predsvm)
confusionMatrix(predsvm,combine_train$final)

##predict SVM on testing set
predsvm <- predict(svm_model,newdata=combine_test)
confusionMatrix(predsvm,combine_test$final)

#SVM Linear Model 
svm.modell <- svm(final ~ . , kernel="linear", cost= 0.01, scale=FALSE, data=combine_train)
summary(svm.modell)

##predict SVM linear for train dataset
svm.predictiontrain <- predict(svm.modell, newdata=combine_train,type="class")
confusionMatrix(svm.predictiontrain,combine_train$final)

##predict SVM linear for test dataset
svm.predictiontest <- predict(svm.modell, newdata=combine_test,type="class")
confusionMatrix(svm.predictiontest,combine_test$final)

#Tune Kernal SVM model first find out what is the best parameter
tunedtrain <- tune(svm,final ~ . ,data=combine_train, kernel="radial", ranges=list(cost=c(0.01,0.1,1,10)))
summary(tunedtrain)

##Plug in the parameter to run the best model
svm.modelr <- svm(final ~ . , kernel="radial", cost= 0.01, scale=FALSE, data=combine_train)
summary(svm.modelr)

#predict SVM radial for train dataset
svm.predictiontrain <- predict(svm.modelr, data=combine_train,type="class")
confusionMatrix(svm.predictiontrain,combine_train$final)

#predict SVM radial for test dataset
svm.predictiontest <- predict(svm.modelr, data=combine_test,type="class")
confusionMatrix(svm.predictiontest,combine_test$final)

