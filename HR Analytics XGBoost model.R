                      # Bismillah - E - Rahman - Ur -Rahim
                      # HR Analytics Competition on Analytics Vidhya

# Library Section
suppressPackageStartupMessages(suppressMessages(suppressWarnings(library(dplyr))))
suppressPackageStartupMessages(suppressMessages(suppressWarnings(library(ggplot2))))
suppressPackageStartupMessages(suppressMessages(suppressWarnings(library(InformationValue))))
suppressPackageStartupMessages(suppressMessages(suppressWarnings(library(lattice))))
suppressPackageStartupMessages(suppressMessages(suppressWarnings(library(caret))))
suppressPackageStartupMessages(suppressMessages(suppressWarnings(library(MLmetrics))))
suppressPackageStartupMessages(suppressMessages(suppressWarnings(library(magrittr))))
suppressPackageStartupMessages(suppressMessages(suppressWarnings(library(xgboost))))
suppressPackageStartupMessages(suppressMessages(suppressWarnings(library(Matrix))))

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
train = train[,c(13,1:12)]
test = test[,c(13,1:12)]
# XGBoost model tried 

# create one hot encoding
trainm= sparse.model.matrix(is_promoted~.-1,data=train)
head(trainm) 
train_label=train[,"is_promoted"]  
train_matrix = xgb.DMatrix(as.matrix(trainm), label=train_label)  
  
testm= sparse.model.matrix(is_promoted~.-1,data=test)
head(testm) 
test_label=test[,"is_promoted"]  
test_matrix = xgb.DMatrix(as.matrix(testm), label=test_label)  

# Parameters for XGBoost
nc=length(unique(train_label))
nc
xgb_params = list("objective"="multi:softprob",
                  "eval_metric"="mlogloss",
                  "num_class"=nc+1)

  
watchlist = list(train = train_matrix, test= test_matrix)

# Extreme Gradiant boosting model
set.seed(123)
mod1 = xgb.train(params= xgb_params,
                 data= train_matrix,
                 nrounds = 608,
                 watchlist = watchlist,
                 eta = 0.05,
                 gamma = 0,
                 max.depth=4,
                 subsample=0.5,
                 colsample_bytree=1,
                 missing=NA)


mod1
e= data.frame(mod1$evaluation_log)
plot(e$iter,e$train_mlogloss,col="blue")
lines(e$iter,e$test_mlogloss, col="red")

min(e$test_mlogloss)
e[e$test_mlogloss=="0.160944",]

# Feature Importance information
imp = xgb.importance(colnames(train_matrix),model =mod1)
print(imp)
xgb.plot.importance(imp)


prediction=predict(mod1,newdata=test_matrix)
head(prediction)
library(dplyr)
pred= matrix(prediction, nrow=nc, ncol=length(prediction)/(nc+1))%>%
      t()%>%
      data.frame()%>%
      mutate(label=test_label,max_prob= max.col(.,"last")-1)

head(pred)
table(Prediction=pred$max_prob,Actual= pred$label)
F1_Score(pred$max_prob,pred$label, positive = 1)

library(MLmetrics)
library(caret)
confusionMatrix((pred$max_prob),(test$is_promoted),positive = "1")

# Predicting on test data of analytics vidhya

val$is_promoted = 0
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
val = val[,c(-1)]
val = val[,c(13,1:12)]
valm= sparse.model.matrix(is_promoted~.-1,data=val)
head(valm) 
val_label=val[,"is_promoted"]  
val_matrix = xgb.DMatrix(as.matrix(valm), label=val_label)  

prediction=predict(mod1,newdata=val_matrix)
head(prediction)
library(dplyr)
pred= matrix(prediction, nrow=nc, ncol=length(prediction)/(nc+1))%>%
  t()%>%
  data.frame()%>%
  mutate(label=val_label,max_prob= max.col(.,"last")-1)

head(pred)
val = read.csv("test.csv")
submission = cbind(val[,1],pred$max_prob)

write.csv(submission,"Submission.csv" ,row.names = FALSE)



