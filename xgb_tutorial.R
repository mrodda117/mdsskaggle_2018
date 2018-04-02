library(plyr)
library(dplyr)
library(xgboost)
library(tidyr)
library(dummies)


setwd("C:/Users/steve/Downloads/kaggle2018")



### SECTION ONE ###
#Combine test and train into one feature set


test <- read.csv("test.csv", stringsAsFactors = FALSE)
train <- read.csv("train.csv", stringsAsFactors = FALSE)

df <- rbind.fill(train, test)

### SECTION TWO ###
#Feature Engineering

#We need to combibe year and month to get a better date attribute
#Use the lazy way of appending month as a decimal place to year attribute

df$better_date <- df$yearOfRegistration + (df$monthOfRegistration / 12)

#Dummify/OneHotEncode the categorical variables we want
df <- df %>% dummy.data.frame( c("gearbox", "fuelType", "brand", "notRepairedDamage", "vehicleType", "model")) 


###SECTION THREE
#Get the features we want along with the target variable (price)
train_df <- df %>% select(-yearOfRegistration, -monthOfRegistration, -postalCode) %>%
  filter(!is.na(price)) %>% apply(2, as.numeric) %>% as.data.frame()
  

#convert to xgb matrix
xgb_mat <- xgb.DMatrix(data = as.matrix(select(train_df, -price)), 
                       label = train_df$price)
 
#Create our parameter input
param_list = list(eta = 0.3,
                  gamma = 2,
                  max_depth = 5,
                  min_child_weight = 4,
                  max_delta_step = 0,
                  subsample = 1,
                  colsample_bytree = 1,
                  colsample_bylevel = 1)

set.seed(117)
#Do Cross Validation to select best model/parameters
cv_model <- xgb.cv(xgb_mat,
                   objective = "reg:linear",
                   eval_metric = "rmse",
                   params = param_list,
                   nrounds = 10000,
                   nfold = 4,
                   early_stopping_rounds = 20)

#With default parameters this yields a test score of 2923.
rmse_score <- as.integer(cv_model$evaluation_log$test_rmse_mean[cv_model$best_iteration])

#Train the actual xgb model
#Use the same parameters and the best iteration of our cross validation model
xgb_model <- xgb.train(xgb_mat,
                       objective = "reg:linear",
                       eval_metric = "rmse",
                       params = param_list,
                       nrounds = cv_model$best_iteration)
                       
#Look at the most important attributes of the model
xgb.importance(feature_names = colnames(select(train_df, -price)), model = xgb_model)
#With default parameters and our chosen features we have date, power and kilometers the top3 most important features.
#Porsche has the most gain in terms of brand.


###SECTION FOUR###
#Create predictions.
#We want all the observations with NA as price 
predict_df <- df %>% select(-id,-yearOfRegistration, -monthOfRegistration, -postalCode) %>%
  filter(is.na(price)) %>% select(-price)

#Get the vector of predictions and convert it into dataframe
predictions <- predict(xgb_model, as.matrix(predict_df)) 

upload <- data.frame(id = 1:length(predictions), price = predictions)

write.csv(upload, paste0("upload6_", rmse_score, ".csv"), row.names = FALSE)