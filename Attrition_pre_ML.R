#read the desired file
MYdataset <-read.csv("D:/1/data.csv")

#str f() will provide the meta data of the dataset it will describe the data type and overview of the data set
str(MYdataset) 

# summary f() provide the insight of the data set mainly it will let us know about the categorical(factors) data
# and central tendency like (mean,median quartiles) for numerical data.
summary(MYdataset)

#preprocessing the data

#It is possible that not all variables are correlated with the label, feature selection is therefore performed to filter out the most relevant ones.
#As the data set is a mix of both numerical and categorical variables,
# A good way to select feature is by training a model and then rank the variable importance so as to select the most salient ones.
# Here I am using decision tree model which come under rpart library to rank important variable.
library(rpart)

dt<-rpart( Attrition~.,data=MYdataset,control=rpart.control(minsplit = 10))

dt$variable.importance
library(caret)
#other way of doing it by Droping the the columns with no variability.
drop_var<-names(MYdataset[, nearZeroVar(MYdataset)])
drop_var #it will show the variable names with zero variability

#as we can see Over18, EmployeeCount and StandardHours having 0 importance and MonthlyIncome, Overtime, Daily Income has more importance

#Cleaning the data

#Drop Over18 as there is no variability, all are Y.
#Drop EmployeeCount as there is no variability, all are 1.
#Drop StandardHours as there is no variability, all are 80.
#Drop EmployeeNumber becouse they are just assigned numers to each Employee
MYdataset$Over18 <- NULL
MYdataset$EmployeeCount <- NULL
MYdataset$StandardHours <- NULL
MYdataset$EmployeeNumber<-NULL


# convert certain integer variable to factor variable  as they make more sense as factor data type
conv_fact<- c("Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel", "JobSatisfaction", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel","TrainingTimesLastYear","WorkLifeBalance")
MYdataset[, conv_fact] <- lapply((MYdataset[, conv_fact]), as.factor)



#visualization of data to get more insights about the dataset

#number of attritions from data
table(MYdataset$Attrition)
ggplot(MYdataset, aes(x=Attrition, fill=Attrition)) + geom_bar()
#it will provide the count of attritions from data

#As per the rpart model, Overtime, MonthlyIncome, TotalWorkingYears, HourlyRate, JobRole and Age are the most important factors influencing the attrition rates. Let's explore these variables.

table(MYdataset$OverTime, MYdataset$Attrition)
ggplot(MYdataset, aes(OverTime, ..count.., fill = factor(Attrition))) + geom_bar(position="dodge")
#Overall 28% of the employees are putting overtime. The percentage of attrition amongst those putting in overtime is close to 44% (254/578) vs 11% (220/1888) for those not putting in overtime. 
#Thus Overtime is contributing towards attrition.

summary(MYdataset$MonthlyIncome)
MnthlyIncome <- cut(MYdataset$MonthlyIncome, 10, include.lowest = TRUE, labels=c(1,2,3,4,5,6,7,8,9,10))
ggplot(MYdataset, aes(MnthlyIncome, ..count.., fill = factor(Attrition))) + geom_bar(position="dodge")
#The attrition in absolute terms decreases as the salary increases, thus lower salary is contributing towards attrition.

summary(MYdataset$TotalWorkingYears)
TtlWkgYrs <- cut(MYdataset$TotalWorkingYears, 10, include.lowest = TRUE)
ggplot(MYdataset, aes(TtlWkgYrs, ..count.., fill = factor(Attrition))) + geom_bar(position="dodge")
#Atrrition decreases as the total number of working years increase. After an employee has spent 8-12 years in the company, his chances of attrition decrease.


summary(MYdataset$HourlyRate)
HrlyRate<- cut(MYdataset$HourlyRate, 7, include.lowest = TRUE)
ggplot(MYdataset, aes(HrlyRate, ..count.., fill = factor(Attrition))) + geom_bar(position="dodge")
#There is no pattern in attrition that can be seen in the context of hourly rates.


table(MYdataset$JobRole, MYdataset$Attrition)
ggplot(MYdataset, aes(JobRole, ..count.., fill = factor(Attrition))) + geom_bar(position="dodge")
# In absolute terms, Laboratory Technicians followed by the Sales Executives are contributing the maximum towards attrition. In percentage terms, Sales Representative are far ahead at 66%.


summary(MYdataset$Age)
A_g_e <- cut(MYdataset$Age, 8, include.lowest = TRUE)
ggplot(MYdataset, aes(A_g_e, ..count.., fill = factor(Attrition))) + geom_bar(position="dodge")
#The age group of 18-23 contributes the maximum to attrition in percentage terms. Post 34, attrition shows a downward trend.



#Hypothesis
#Different job roles has differsnt atrrition means Job type also plays an important role in Attrition.
#Overtime is the single most important factor contributing towards attrition.
#Young people with lesser years at the company has more attrition rate.
#Lower the income Highest is the Attrition rate
#we will now apply different Machine learning techniques to predict the Attrition
# And Check Which ML techniques works better in prediction.

#Create Training and Testing Sets.
library(dplyr)
ModelData<-sample_frac(MYdataset, 0.75)
sid<-as.numeric(rownames(ModelData)) # because rownames() returns character
ValidateData<-MYdataset[-sid,]


#As per the rpart model, Overtime, MonthlyIncome, TotalWorkingYears, HourlyRate, JobRole and Age are the most important factors influencing the attrition rates. Let's explore these variables.
#only taking the variables of interest to speed up the process and reduce the complexity you can add any variable if you think it may affect on model
Subject <- c("OverTime","MonthlyIncome","TotalWorkingYears","HourlyRate","JobRole","Age")
Objective  <- "Attrition"

TrainingData<-ModelData[c(Subject, Objective)]
ValidationData<-ValidateData[c(Subject, Objective)]


#before building the model now we are creating the function for ROC(Receiver operating characteristic) AUC (Area Under Curve) value if we want to know
# more details about the model or whether we want to tune the model or not

library(ROCR)
### AUC plot function
fun.aucplot <- function(pred, obs, title){
  # Run the AUC calculations
  ROC_perf <- performance(prediction(pred,obs),"tpr","fpr")
  ROC_sens <- performance(prediction(pred,obs),"sens","spec")
  ROC_auc <- performance(prediction(pred,obs),"auc")
  # Spawn a new plot window (Windows OS)
  graphics.off(); x11(h=6,w=6)
  # Plot the curve
  plot(ROC_perf,colorize=T,print.cutoffs.at=seq(0,1,by=0.1),lwd=3,las=1,main=title)
  abline(a=0,b=1)
  # Add some statistics to the plot
  text(1,0.15,labels=paste("AUC = ",round(ROC_auc@y.values[[1]],digits=2),sep=""),adj=1)
}

#============================================================
# Classification Models
# Support vector machine. 
library(e1071)
model <- svm(Attrition ~ .,data=TrainingData)#SVM model on data set
summary(model)
predict_val <- predict(model,ValidationData[1:6]) #predicting the values

library(caret)
#confusionMatrix Calculates a cross-tabulation of observed and predicted classes with associated statistics.
result_Test <- confusionMatrix(predict_val,ValidationData$Attrition)
result_Test

#As we can observe from the Confusion matricx the False Positive is very high
# now we can proceed further or check the ROC and AUC values and decide the tuning for the model
Validate<-ifelse(ValidationData$Attrition=="Yes",1,0)
predicted<-ifelse(predict_val=="Yes",1,0)
# Run the function
fun.aucplot(predicted,ValidationData$Attrition, "My AUC Plot")

#As we can see the AUC value is only 0.52 and Specificity is also low we need to tune the model

# While tuning SVM we need to sure that high cost and gamma will always do better
# But it eventually overfit the model so we need to be very careful about cost and gamma values
tune.results <- tune(svm,train.x=Attrition~., data=TrainingData,kernel='radial',
                     ranges=list(cost=c(1,5,10), gamma=c(0.1,0.5,1)))

summary(tune.results)

# In the result of summary cost =5, and gamma=1 will give good performance
# or else if you want low  values then cost=5 and gamma =0.5 is best among remaining

model <-svm(Attrition ~ .,data=TrainingData,cost=5,gamma = 1)
summary(model)

predict_val <- predict(model,ValidationData[1:6]) #predicting the values

#confusionMatrix Calculates a cross-tabulation of observed and predicted classes with associated statistics.
result_Test <- confusionMatrix(predict_val,ValidationData$Attrition)
result_Test

#As we can observe from the Confusion matricx the False Positive is very high
# now we can proceed further or check the ROC and AUC values and decide the tuning for the model
Validate<-ifelse(ValidationData$Attrition=="Yes",1,0)
predicted<-ifelse(predict_val=="Yes",1,0)
# Run the function
fun.aucplot(predicted,ValidationData$Attrition, "My AUC Plot")

#As we can see the AUC value is 0.73 which is higher than previous and Specificity is also increased we need to tune the model
#and the Accuracy is 0.8803


#============================================================

# Decision Tree 

# Build the Decision Tree model.



# if dependent var is a factor then method = "class" is used
# other methods are anova, poisson,exp.
# parms=optional parameters for the splitting function.
# The splitting index can be gini or information.
# control=a list of options that control details of the rpart algorithm.
# use surrogate=how to use surrogates in the splitting process. 0 means display only; an observation with a missing value for the primary split rule is not sent further down the tree. other 1,2
# maxsurrogate=the number of surrogate splits retained in the output. If this is set to zero the compute time will be reduced, since approximately half of the computational time (other than setup) is used in the search for surrogate splits.
library(rpart)
#By default, rpart uses gini impurty to select splits when performing classification.
#You can use information gain instead by specifying it in the parms parameter.
model_DT <- rpart(Attrition ~ .,
                 data=TrainingData,
                 method="class",
                 parms=list(split="information"),
                 control=rpart.control(usesurrogate=0, 
                                       maxsurrogate=0))
summary(model_DT, direction="forward")
# As we can see in the CP table after 7 splits the error got reduced to 0.807 which is pretty low which means model is performing well

#testing the model on test data
pred_Test_DT<-predict(model_DT,ValidationData[1:6], type="class")

result_Test <- confusionMatrix(ValidationData$Attrition, pred_Test_DT)
result_Test

#Accuracy is 0.8408 for the model

Validate<-ifelse(ValidationData$Attrition=="Yes",1,0)
predicted<-ifelse(pred_Test_DT=="Yes",1,0)
# Run the function
fun.aucplot(predicted,Validate, "My AUC Plot")

#As we can see the AUC value is 0.6 
#and the Accuracy is 0.8544 
#============================================================
# Classification Models
library(randomForest)


# ntree=Number of trees to grow. This should not be set to too small a number, to ensure that every input row gets predicted at least a few times.
# mtry=Number of variables randomly sampled as candidates at each split. Note that the default values are different for classification (sqrt(p) where p is number of variables in x) and regression (p/3)
# importance=Should importance of predictors be assessed?
# Replace=Should sampling of cases be done with or without replacement?
# control=a list of options that control details of the rpart algorithm.
# use surrogate=how to use surrogates in the splitting process. 0 means display only; an observation with a missing value for the primary split rule is not sent further down the tree. other 1,2
# maxsurrogate=the number of surrogate splits retained in the output. If this is set to zero the compute time will be reduced, since approximately half of the computational time (other than setup) is used in the search for surrogate splits.
model_RF <- randomForest(Attrition ~ .,
                     data=TrainingData,
                     ntree=500,
                     mtry=2,
                     importance=TRUE,
                     replace=FALSE)

# Generate a textual view of the Random Forest
model_RF
summary(model_RF)
plot(model_RF, main="")
legend("topright", c("OOB", "0", "1"), text.col=1:6, lty=1:3, col=1:3)
title(main="Error Rates Random Forest")

#testing the model on test data
pred_Test_RF<-predict(model_RF,ValidationData[1:6], type="class")
result_Test <- confusionMatrix(ValidationData$Attrition, pred_Test_RF)
result_Test


#Accuracy for Random forest is 0.9592   which is high compare to other model which have not exceeded 90%.

Validate<-ifelse(ValidationData$Attrition=="Yes",1,0)
predicted<-ifelse(pred_Test_RF=="Yes",1,0)

fun.aucplot(predicted,Validate, "My AUC Plot")

#As we can see the AUC value is 0.87
#and the Accuracy is 0.9551 
#=====================================
#ADABOOST
library(ada)
#cp=complexity parameter. Any split that does not decrease the overall lack of fit by a factor of cp is not attempted. 
#xval=number of cross-validations
model_ADA <- ada(Attrition ~ .,
             data=TrainingData,
             control=rpart::rpart.control(maxdepth=30,
                                          cp=0.010000,
                                          minsplit=20,
                                          xval=10),
             iter=50)
# Print the results of the modelling.
print(model_ADA)
#testing the model for overfitting on training data
pred_Test_ADA<-predict(model_ADA,ValidationData)


result <- confusionMatrix(ValidationData$Attrition, pred_Test_ADA)
result


#Accuracy of the model is 0.8585  
Validate<-ifelse(ValidationData$Attrition=="Yes",1,0)
predicted<-ifelse(pred_Test_ADA=="Yes",1,0)
# Run the function
fun.aucplot(predicted,Validate, "My AUC Plot")
#As we can see the AUC value is 0.62
#and the Accuracy is 0.8667    



# We have applied 4 Machine learning Techniques on Attrition data and the accuracy of each model is
#SVM			88%
#Decision Tree	85% 
#Random Forest	95%
#ADABOOST		86%

# We can say that from all the ML algorithms Random Forest works better on Attrition data.