                                # Bismillah - E - Rahman - Ur -Rahim
                                # HR Analytics Competition on Analytics Vidhya

# Library Section
suppressPackageStartupMessages(suppressMessages(suppressWarnings(library(e1071))))
suppressPackageStartupMessages(suppressMessages(suppressWarnings(library(esquisse))))
suppressPackageStartupMessages(suppressMessages(suppressWarnings(library(dplyr))))
suppressPackageStartupMessages(suppressMessages(suppressWarnings(library(ggplot2))))
suppressPackageStartupMessages(suppressMessages(suppressWarnings(library(InformationValue))))
suppressPackageStartupMessages(suppressMessages(suppressWarnings(library(lattice))))
suppressPackageStartupMessages(suppressMessages(suppressWarnings(library(caret))))
suppressPackageStartupMessages(suppressMessages(suppressWarnings(library(MLmetrics))))

#Set Working Directory
getwd()
setwd("E:\\WorkStation Laptop\\Learning Material\\Competitions\\HR Analytics")

#Read train & Test Files
data = read.csv("train.csv")
val = read.csv("test.csv")

# Explore your data
summary(data)
unique(data$department)
barplot(table(data$department), xlab="Department",ylab="No of Employee",main="Department wise bifurcation")
#esquisse::esquisser()

library(dplyr)
library(ggplot2)

data %>%
        filter(!(education %in% "")) %>%
        ggplot() +
        aes(x = education, y = avg_training_score, fill = is_promoted) +
        geom_boxplot(shape = "circle") +
        scale_fill_hue(direction = 1) +
        theme_minimal()

data %>%
        filter(!(education %in% "")) %>%
        ggplot() +
        aes(x = department, y = avg_training_score, fill = is_promoted) +
        geom_boxplot(shape = "circle") +
        scale_fill_hue(direction = 1) +
        theme_minimal()


# in all combinations gender doesnt seem to have any impact on promotion
# Department wise  it looks that if previous year rating is low in sales then promotion is unlikely
# Analytics & Sales seem to have higher training score because of which thier promotions are more likely


# Reshaping the data
data$is_promoted = as.factor(data$is_promoted)
data$department = as.factor(data$department)
data$region = as.factor(data$region)
data$education = as.character(data$education)
data$education[data$education == ""] <- "Not Known"
data$education = as.factor(data$education)
data$gender=as.factor(data$gender)
data$recruitment_channel = as.factor(data$recruitment_channel)
data$previous_year_rating = as.character(data$previous_year_rating)
data$previous_year_rating[is.na(data$previous_year_rating)]<-0
data$previous_year_rating = as.ordered(data$previous_year_rating)
data$employee_id = as.character(data$employee_id)
summary(data)

# creating a simple logistics regrssion model and checking for F 1 score

set.seed(222)
data1= data
data = data1[,c(2:14)]
t=sample(1:nrow(data),0.7*nrow(data))
train=data[t,]
test=data[-t,]


# Creating Logistics Regression model

mod1 <- glm(is_promoted ~ ., family="binomial", data=train)
summary(mod1)

train$score=predict(mod1,newdata=train,type = "response")
prediction<-ifelse(train$score>=0.29,1,0)
prediction= as.factor(prediction)

AUC(prediction, train$is_promoted) # 0.616

# F1 Score
f1_scores <- sapply(seq(0.01, 0.99, .01), function(thresh) F1_Score(train$is_promoted, ifelse(train$score >= thresh, 1, 0), positive = 1))
which.max(f1_scores) #29
F1_Score(train$is_promoted,prediction, positive=1)

confusionMatrix(prediction,train$is_promoted,positive = "1")

plotROC(actuals = train$is_promoted,predictedScores = as.numeric(fitted(mod1)))
ks_plot(actuals = train$is_promoted,predictedScores = as.numeric(fitted(mod1)))
ks_stat(actuals = train$is_promoted,predictedScores = as.numeric(fitted(mod1)))


test$score= predict(mod1, test, type="response") 
prediction<-ifelse(test$score>=0.29,1,0)
prediction= as.factor(prediction)
confusionMatrix(prediction,test$is_promoted,positive = "1")
AUC(prediction, test$is_promoted) # 0.611
F1_Score(test$is_promoted,prediction, positive=1)


# Trying for Step wise model for model iteration
summary(mod1)
step(mod1)

mod1 <- glm(is_promoted ~ department + region + education + 
                    recruitment_channel + no_of_trainings + age + previous_year_rating + 
                    length_of_service + KPIs_met..80. + awards_won. + avg_training_score + 
                    score, family = "binomial", data = train)

train$score=predict(mod1,newdata=train,type = "response")
prediction<-ifelse(train$score>=0.31,1,0)
prediction= as.factor(prediction)

AUC(prediction, train$is_promoted) # 0.69

# F1 Score
f1_scores <- sapply(seq(0.01, 0.99, .01), function(thresh) F1_Score(train$is_promoted, ifelse(train$score >= thresh, 1, 0), positive = 1))
which.max(f1_scores) #29
F1_Score(train$is_promoted,prediction, positive=1)

confusionMatrix(prediction,train$is_promoted,positive = "1")

plotROC(actuals = train$is_promoted,predictedScores = as.numeric(fitted(mod1)))
ks_plot(actuals = train$is_promoted,predictedScores = as.numeric(fitted(mod1)))
ks_stat(actuals = train$is_promoted,predictedScores = as.numeric(fitted(mod1)))


test$score= predict(mod1, test, type="response") 
prediction<-ifelse(test$score>=0.31,1,0)
prediction= as.factor(prediction)
confusionMatrix(prediction,test$is_promoted,positive = "1")
Accuracy(prediction,test$is_promoted)
AUC(prediction, test$is_promoted) # 0.611
F1_Score(test$is_promoted,prediction, positive=1)



# Support Vector machine all combinations being tried 

mod1 = svm(is_promoted~., 
               data = train,
               type= 'C-classification' , 
               kernel="radial", 
               cost = 1, 
               gamma=0.5)


prediction=predict(mod1,newdata=train,type = "response")
prediction= as.factor(prediction)

AUC(prediction, train$is_promoted) # 0.69

# F1 Score
F1_Score(train$is_promoted,prediction, positive=1)

confusionMatrix(prediction,train$is_promoted,positive = "1")

plotROC(actuals = train$is_promoted,predictedScores = as.numeric(fitted(mod1)))
ks_plot(actuals = train$is_promoted,predictedScores = as.numeric(fitted(mod1)))
ks_stat(actuals = train$is_promoted,predictedScores = as.numeric(fitted(mod1)))

prediction= predict(mod1, test, type="response") 
prediction= as.factor(prediction)
confusionMatrix(prediction,test$is_promoted,positive = "1")
Accuracy(prediction,test$is_promoted)
AUC(prediction, test$is_promoted) # 0.611
F1_Score(test$is_promoted,prediction, positive=1)


# Predicting on test data of analytics vidhya

val$is_promoted = NA
val$department = as.factor(val$department)
val$region = as.factor(val$region)
val$education = as.character(val$education)
val$education[val$education == ""] <- "Not Known"
val$education = as.factor(val$education)
val$gender=as.factor(val$gender)
val$recruitment_channel = as.factor(val$recruitment_channel)
val$previous_year_rating = as.character(val$previous_year_rating)
val$previous_year_rating[is.na(val$previous_year_rating)]<-0
val$previous_year_rating = as.ordered(val$previous_year_rating)
val$employee_id = as.character(val$employee_id)
summary(val)

# Predicting on validation data set 
prediction= predict(mod1, val, type="response") 
prediction= as.factor(prediction)
val$is_promoted = prediction
submission = val[,c(1,14)]

write.csv(submission,"Submission.csv")



