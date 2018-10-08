#Practical Machine Learning - Course Project

##Introduction 
For this project, we are given data from accelerometers on the belt, forearm, arm, and dumbell of 6 research study participants. Our training data consists of accelerometer data and a label identifying the quality of the activity the participant was doing. Our testing data consists of accelerometer data without the identifying label. Our goal is to predict the labels for the test set observations.

Below is the code I used when creating the model, estimating the out-of-sample error, and making predictions. I also include a description of each step of the process.

Data Preparation
I load the caret package, and read in the training and testing data:

```{r}
library(caret)

train <- read.csv("C:\\Users\\mariat\\Downloads\\pml-training.csv")
test <- read.csv("C:\\Users\\mariat\\Downloads\\pml-testing.csv")
```

## Pre-processing
We need to pre-process our data before we can use it for modeling. Let's check if the data has any missing values:

```{r}
sum(is.na(train))
```


I am now going to reduce the number of features by removing variables with nearly zero variance, and variables that don't make intuitive sense for prediction. Note that I decide which ones to remove by analyzing train.Because I want to be able to estimate the out-of-sample error, I randomly split the full training data (train_processed) into a smaller training set (train1) and a validation set (test1):


```{r}
# remove variables with nearly zero variance
inTrain<-createDataPartition(y=train$classe,p=0.7,list=FALSE)
train1<-train[inTrain,]
test1<-train[-inTrain,]
nzvtrain <- nearZeroVar(train1)
nzvtest<-nearZeroVar(test1)
train_processed <- train1[, -nzvtrain]
test_processed<-test1[,-nzvtest]

# remove variables that are almost always NA
mostlyNAtrain <- sapply(train_processed, function(x) mean(is.na(x))) > 0.95
train_processed <- train_processed[, mostlyNAtrain==F]
mostlyNAtest <- sapply(test_processed, function(x) mean(is.na(x))) > 0.95
test_processed <- test_processed[, mostlyNAtest==F]


# remove variables that don't make intuitive sense for prediction (X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp), which happen to be the first five variables
train_processed <- train_processed[, -(1:5)]
test_processed<-test_processed[,-(1:5)]
```


##Prediction with Decision Trees
```{r}
set.seed(1222)
library(rpart)
fitRpart<-train(classe ~.,data=train_processed,method='rpart')

plot(fitRpart$finalModel,uniform=TRUE,main='Classification Trees')
text(fitRpart$finalModel,use.n=TRUE,all=TRUE,cex=.8)

predRpart<-predict(fitRpart,test_processed)
cM<-confusionMatrix(predRpart,test_processed$classe)
cM
plot(cM$table,col=cM$byClass,main=paste('Decision Tree Confision Matrix: Accuracy=',round(cM$overall['Accuracy'],4)))
```


We can see tha the accuracy rate is very low and, therefore, the out-of-sample-error is about 0.4.


##Random Forest Prediction
I fit the model on train_processed, and instruct the "train" function to use 3-fold cross-validation to select optimal tuning parameters for the model.

```{r}
# instruct train to use 3-fold CV to select optimal tuning parameters
fitControl <- trainControl(method="cv", number=3, verboseIter=F)

# fit model on ptrain1
fit <- train(classe ~ ., data=train_processed, method="rf", trControl=fitControl)


# print final model to see tuning parameters it chose
fit$finalModel
```


##Model Evaluation and Selection
Now, I use the fitted model to predict the label ("classe") in test_processed, and show the confusion matrix to compare the predicted versus the actual labels:

```{r}
# use model to predict classe in validation set (ptrain2)
preds <- predict(fit, newdata=test_processed)

# show confusion matrix to get estimate of out-of-sample error
confusionMatrix(test_processed$classe, preds)
```

The accuracy is 99.7%, thus my predicted accuracy for the out-of-sample error is 0.2%.

This is an excellent result, so rather than trying additional algorithms, I will use Random Forests to predict on the test set.

##Re-training the Selected Model
Before predicting on the test set, it is important to train the model on the full training set (train), rather than using a model trained on a reduced training set (train_procesesd), in order to produce the most accurate predictions. 

```{r}

# re-fit model using full training set (train)
nzv<- nearZeroVar(train)

train1 <- train[, -nzv]
test1<-test[,-nzv]

# remove variables that are almost always NA
mostlyNAtrain <- sapply(train1, function(x) mean(is.na(x))) > 0.95
train1 <- train1[, mostlyNAtrain==F]
mostlyNAtest <- sapply(test1, function(x) mean(is.na(x))) > 0.95
test1 <- test1[, mostlyNAtest==F]


# remove variables that don't make intuitive sense for prediction (X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp), which happen to be the first five variables
train1 <- train1[, -(1:5)]
test1<-test1[,-(1:5)]
fitControl <- trainControl(method="cv", number=3, verboseIter=F)
fit <- train(classe ~ ., data=train1, method="rf", trControl=fitControl)
```

Making Test Set Predictions
Now, I use the model fit on train to predict the label for the observations in test, and write those predictions to individual files:

```{r}
# predict on test set
preds <- predict(fit, newdata=test1)

# convert predictions to character vector
preds <- as.character(preds)

# create function to write predictions to files
pml_write_files <- function(x) {
    n <- length(x)
    for(i in 1:n) {
        filename <- paste0("problem_id_", i, ".txt")
        write.table(x[i], file=filename, quote=F, row.names=F, col.names=F)
    }
}

# create prediction files to submit
pml_write_files(preds)

```

