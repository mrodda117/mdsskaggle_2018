library(plyr)
library(dplyr)
library(xgboost)
library(tidyr)
library(dummies)

### SECTION ONE - COMBINING###
#Combine test and train into one feature set
test <- read.csv("test.csv", stringsAsFactors = FALSE)
train <- read.csv("train.csv", stringsAsFactors = FALSE)

df <- rbind.fill(train, test)
rm(test, train) #Remove to save memory

### SECTION TWO - FEATURE ENGINEERING & CLEANING ###
#Feature Engineering

#We need to combibe year and month to get a better date attribute
#Use the lazy way of appending month as a decimal place to year attribute
df$better_date <- df$yearOfRegistration + (df$monthOfRegistration / 12)

#Lets keep things simple
#convert the categorical variables "brand" and "notRepairedDamage" into dummy variables
#The dummies package is a lifesaver for this conversion process
df <- df %>% dummy.data.frame( c("brand", "notRepairedDamage")) 


###SECTION THREE - MODELLING###
#Get the features we want along with the target variable (price)
#For this simple model, we are using the vehicle brand, if its had repairs, kilometers and power to predict price
train_df <- df %>% 
  select(-yearOfRegistration, -monthOfRegistration, -postalCode, -gearbox, -fuelType, -vehicleType, -model, -id) %>%
  filter(!is.na(price)) %>%
  apply(2, as.numeric) %>% #Sometimes xgboost doesnt like the variable type. convert everything to numeric because its linear regression
  as.data.frame()
  

#convert to xgb matrix
xgb_mat <- xgb.DMatrix(data = as.matrix(select(train_df, -price)), 
                       label = train_df$price)
 
#Create our parameter input
#These are the default inputs, get experimenting and see how xgb.cv changes!
param_list = list(eta = 0.3,
                  gamma = 0,
                  max_depth = 6,
                  min_child_weight = 1,
                  max_delta_step = 0,
                  subsample = 1,
                  colsample_bytree = 1,
                  colsample_bylevel = 1)

set.seed(117)
#Do Cross Validation to select best model/parameters
cv_model <- xgb.cv(data = xgb_mat,
                   objective = "reg:linear",       #Linear Regression
                   eval_metric = "rmse",           #Root Mean Square Error
                   params = param_list,
                   nrounds = 10000,                #Set to a high number, early_stopping_rounds is how we terminate the cv
                   nfold = 4,                      #4 K-Fold CV
                   early_stopping_rounds = 20)     #Stop after 20 rounds of not improving test score

#With default parameters this yields a test score of...
rmse_score <- as.integer(cv_model$evaluation_log$test_rmse_mean[cv_model$best_iteration])

#Train the actual xgb model
#Use the same parameters and the best number of iterations from our cross validation model
xgb_model <- xgb.train(xgb_mat,
                       objective = "reg:linear",
                       eval_metric = "rmse",
                       params = param_list,
                       nrounds = cv_model$best_iteration)
                       
#Look at the most important attributes of the model
xgb.importance(feature_names = colnames(select(train_df, -price)), model = xgb_model)
#With default parameters and our chosen features we have date, power and kilometers the top3 most important features.
#Porsche has the most gain in terms of brand and is more "important" than if repair status.


###SECTION FOUR###
#Create predictions.
#We want all the observations with NA as price 
predict_df <- df %>% 
  select(-yearOfRegistration, -monthOfRegistration, -postalCode, -gearbox, -fuelType, -vehicleType, -model, -id) %>%
  filter(is.na(price)) %>%
  select(-price)

#Get the vector of predictions and convert it into dataframe
predictions <- predict(xgb_model, as.matrix(predict_df)) 

upload <- data.frame(id = 1:length(predictions), price = predictions)

write.csv(upload, paste0("upload1_starterscript_", rmse_score, ".csv"), row.names = FALSE)