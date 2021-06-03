#Data and Web Mining Project
#Carl Flynn x17347726
#Cian Larkin x17453136

#Set working directory
setwd("C:/Users/CarlF/OneDrive/Desktop/Semester 2/Data Mining/Project")

#Install Packages
library(tidyverse)
library(reshape)

#Loading Data
HeartFailure <- read.csv("heart_failure_clinical_records_dataset.csv", header = TRUE)

#Starting Classification
y <- HeartFailure$DEATH_EVENT

y <- factor(y, levels = c(0,1), labels = c("No", "Yes"))
table(y)

prop.table(table(y))

set.seed(1337)
index <- sample(1:length(y), length(y) * .33, replace=FALSE)
testing <- y[index]

HeartModel <- rep("No", length(testing))

HeartModel <- factor(HeartModel, levels = c("No", "Yes"), labels = c("No", "Yes"))

table(testing, HeartModel)

(HeartAccuracy <- 1 - mean(HeartModel != testing))

prop.table(table(testing))

Heart <- c()
for (i in 1:1000) {
  index <- sample(1:length(y), length(y) * .25, replace=FALSE)
  testing <- y[index]
  HeartModel <- round(runif(length(testing), min=0, max=1))
  HeartModel <- factor(HeartModel, levels = c(0,1), labels = c("No", "Yes"))
  Heart[i] <- 1 - mean(HeartModel != testing)
}
results <- data.frame(Heart)
names(results) <- c("Heart Failure Accuracy")
summary(results)

ggplot(melt(results), mapping = aes (fill = variable, x = value)) + geom_density (alpha = .5)


#Diabetes
Diabetes_Death <- HeartFailure[, c("diabetes", "DEATH_EVENT")]
Diabetes_Death$DEATH_EVENT <- factor(Diabetes_Death$DEATH_EVENT, levels = c(0,1), labels = c("No", "Yes"))

index <- sample(1:dim(Diabetes_Death)[1], dim(Diabetes_Death)[1] * .66, replace=FALSE)
trainingDiabetes <- Diabetes_Death[index, ]
testingDiabetes <- Diabetes_Death[-index, ]
table(trainingDiabetes$DEATH_EVENT, trainingDiabetes$diabetes)

predictDeath <- function(data) {
  model <- rep("No", dim(data)[1])
  model[data$diabetes == '1'] <- "Yes"
  return(model)
}

diabetes <- c()
for (i in 1:1000) {
  index <- sample(1:dim(Diabetes_Death)[1], dim(Diabetes_Death)[1] * .66, replace=FALSE)
  testingDiabetes <- Diabetes_Death[-index, ]
  DiabetesModel <- predictDeath(testingDiabetes)
  diabetes[i] <- 1 - mean(DiabetesModel != testingDiabetes$DEATH_EVENT)
}
results$`Diabetes Accuracy` <- diabetes
names(results) <- c("HeartFailure", "Diabetes")
boxplot(results)

#High Blood Pressure
HBP_Death <- HeartFailure[, c("high_blood_pressure", "DEATH_EVENT")]
HBP_Death$DEATH_EVENT <- factor(HBP_Death$DEATH_EVENT, levels = c(0,1), labels = c("No", "Yes"))

index <- sample(1:dim(HBP_Death)[1], dim(HBP_Death)[1] * .66, replace=FALSE)
trainingHBP <- HBP_Death[index, ]
testingHBP <- HBP_Death[-index, ]
table(trainingHBP$DEATH_EVENT, trainingHBP$high_blood_pressure)

predictDeath <- function(data) {
  HBPmodel <- rep("No", dim(data)[1])
  HBPmodel[data$high_blood_pressure == '1'] <- "Yes"
  return(HBPmodel)
}

high_blood_pressure <- c()
for (i in 1:1000) {
  index <- sample(1:dim(HBP_Death)[1], dim(HBP_Death)[1] * .66, replace=FALSE)
  testingHBP <- HBP_Death[-index, ]
  HBPModel <- predictDeath(testingHBP)
  high_blood_pressure[i] <- 1 - mean(HBPModel != testingHBP$DEATH_EVENT)
}
results$`High Blood Pressure Accuracy` <- high_blood_pressure
names(results) <- c("HeartFailure", "High Blood Pressure","Diabetes")
boxplot(results)

#Smoking
Smoking_Death <- HeartFailure[, c("smoking", "DEATH_EVENT")]
Smoking_Death$DEATH_EVENT <- factor(Smoking_Death$DEATH_EVENT, levels = c(0,1), labels = c("No", "Yes"))

index <- sample(1:dim(Smoking_Death)[1], dim(Smoking_Death)[1] * .66, replace=FALSE)
smokingTraining <- Smoking_Death[index, ]
smokingTesting <- Smoking_Death[-index, ]
table(smokingTraining$DEATH_EVENT, smokingTraining$smoking)

predictDeath <- function(data) {
  smokingModel <- rep("No", dim(data)[1])
  smokingModel[data$smoking == '1'] <- "Yes"
  return(smokingModel)
}

smoking <- c()
for (i in 1:1000) {
  index <- sample(1:dim(Smoking_Death)[1], dim(Smoking_Death)[1] * .66, replace=FALSE)
  smokingTesting <- Smoking_Death[-index, ]
  smokingModel <- predictDeath(smokingTesting)
  smoking[i] <- 1 - mean(smokingModel != smokingTesting$DEATH_EVENT)
}
results$`Smoking Accuracy` <- smoking
names(results) <- c("HeartFailure","Smoking", "High Blood Pressure","Diabetes")
boxplot(results)

#Evaluating Performance
confusionMatrix(factor(DiabetesModel), testingDiabetes$DEATH_EVENT, positive = "Yes")
confusionMatrix(factor(HBPModel), testingHBP$DEATH_EVENT, positive = "Yes")
confusionMatrix(factor(smokingModel), smokingTesting$DEATH_EVENT, positive = "Yes")

#KNN
library(caret)
library(gmodels)
set.seed(1337)
#Loading Data
HeartFailure <- read.csv("heart_failure_clinical_records_dataset.csv", header = TRUE)

HeartFailure$DEATH_EVENT <- factor(HeartFailure$DEATH_EVENT, levels = c(0,1), labels = c("No", "Yes"))

n <- sapply(HeartFailure, function(x) {is.numeric(x)})
n

numerics <-HeartFailure[, n]
summary(numerics)

normalize <- function(x) { return ((x - min(x)) / (max(x) - min(x))) }
numericsNormal <- normalize(numerics)
summary(numericsNormal)

HeartFailureKNN <- HeartFailure[, !n]
HeartFailureKNN <- cbind(HeartFailureKNN, numericsNormal)

tkNN <- dummy.data.frame(HeartFailureKNN)
summary(tkNN)

Died <- tkNN$HeartFailureKNNYes


index <- createDataPartition(HeartFailure$DEATH_EVENT, p = .66,
                             list = FALSE,
                             times = 1)
kNNTraining <- tkNN[index,-c(1,2)]
kNNTesting <- tkNN[-index,-c(1,2)]

DiedTrain <- tkNN[index,]$HeartFailureKNNYes
DiedTest <- tkNN[-index,]$HeartFailureKNNYes

summary(DiedTrain)

k1 <- round(sqrt(dim(kNNTraining)[1])) #sqrt of number of instances
k2 <- round(sqrt(dim(kNNTraining)[2])) #sqrt of number of attributes
k3 <- 8 #a number between 3 and 10

knn1 <- knn(train = kNNTraining, test = kNNTesting, cl = DiedTrain, k=k1)
knn2 <- knn(train = kNNTraining, test = kNNTesting, cl = DiedTrain, k=k2)
knn3 <- knn(train = kNNTraining, test = kNNTesting, cl = DiedTrain, k=k3)

confusionMatrix(knn1, as.factor(DiedTest))
confusionMatrix(knn2, as.factor(DiedTest))
confusionMatrix(knn3, as.factor(DiedTest))

