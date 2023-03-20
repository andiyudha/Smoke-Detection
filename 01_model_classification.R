## Modeling Smoke Detection to Fire Alarm Prediction

# Smoke detection prediction with dataset from https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset with classification model in R Programming language
# 
# Used for Balikpapan Hackathon 2023



# Load Library ------------------------------------------------------------

# Load library that will use in this model
library(tidyverse)
library(caret)
library(rpart.plot)
library(odbc)
library(RPostgres)
library(corrplot)
library(ggcorrplot)
library(RColorBrewer)
library(hrbrthemes)


# read and wrangling data -------------------------------------------------

# read data for modeling from opendataset 
# read raw data
read_csv("./raw data/smoke_detection_iot.csv") %>%
  mutate(
    index = `...1`,
    UTC = as_datetime(UTC)
  ) %>% 
  select(-`...1`) %>%
  glimpse() -> raw_data

# Missing data
# Check missing data on object dataset

raw_data %>%
  summarise_all(~ sum(is.na(.)))

# Found that no missing dataset in object & we can perform data processing for these dataset

# Dataset consists of variables:

# 1. UTC: Time when experiment was performed
# 2. Temperature[C]: Temperature of surroundings, measured in celcius
# 3. Humidity[%]: Air humidity during the experiment
# 4. TVOC[ppb]: Total Volatile Organic Compounds, measured in ppb (parts per billion)
# 5. eCO2[ppm]: CO2 equivalent concentration, measured in ppm (parts per million)
# 6. Raw H2: The amount of Raw Hydrogen [Raw Molecular Hydrogen; not compensated (Bias, Temperature etc.)] present in surroundings
# 7. Raw Ethanol: The amount of Raw Ethanol present in surroundings
# 8. Pressure[hPa]: Air pressure, Measured in hPa
# 9. PM1.0: Paticulate matter of diameter less than 1.0 micrometer
# 10. PM2.5: Paticulate matter of diameter less than 2.5 micrometer
# 11. NC0.5: Concentration of particulate matter of diameter less than 0.5 micrometer
# 12. NC1.0: Concentration of particulate matter of diameter less than 1.0 micrometer
# 13. NC2.5: Concentration of particulate matter of diameter less than 2.5 micrometer
# 14. CNT: Sample Count. Fire Alarm(Reality) If fire was present then value is 1 else it is 0
# 15. Fire Alarm: 1 means Positive and 0 means Not Positive

glimpse(raw_data)

# data processing ---------------------------------------------------------

# Processing dataset before doing a modeling. Selecting the variables that needed & unselect variable CNT, Sample Count.

raw_data %>%
  select(
    temp_c = `Temperature[C]`,
    humidity = `Humidity[%]`,
    tvoc = `TVOC[ppb]`,
    co2 = `eCO2[ppm]`,
    h2 = `Raw H2`,
    ethanol = `Raw Ethanol`,
    pressure = `Pressure[hPa]`,
    pm1 = PM1.0,
    pm2_5 = PM2.5,
    fire_alarm = `Fire Alarm`
  ) %>%
  mutate(
    fire_alarm = factor(fire_alarm, levels = c(1,0), labels = c("yes", "no"))
  ) %>%
  glimpse() -> df_data

# Correlation plot

df_data %>%
  mutate(
    fire_alarm = case_when(
      fire_alarm == "yes" ~ 1,
      fire_alarm == "no" ~ 0,
    )
  ) %>%
  cor() %>%
  corrplot(method = "color",
           type = "lower",
           tl.col = "black", tl.srt = 1,
           addCoef.col = "black",
           number.cex = 0.6,
           col=brewer.pal(n=8, name="RdYlBu"),
           title = "Correlation variable on Smoke Detection",
           mar=c(1,0,1,1))

# **Insight**
#   1. There is not any high correlation between target feature and other features. Small positive correlation between target **feature** and **Humidity**, **Pressure**. Small negative correlation between target feature and *TVOC*, *Raw Ethanol*.
# 2. High positive correlation between eCO2 and TVOC, PM1.0. Pressure and Humidity. Raw H2 and Raw Ethanol. PM1.0 and eCO2, PM2.5.


# split data --------------------------------------------------------------

# Splitting data into train and test dataset with proportion 80:20, Usually you'll get more accurate models the bigger that dataset you're training on, but more training data also leads to models taking longer to train.
# 
# To split our data, we're going to use the createDataPartition() from the caret package. The function randomly samples the a proportion of the indexes of a vector you pass it. Then you can use those indexes to subset your full dataset into testing and training datasets.

# set random number
set.seed(123)

# splitiing data
train_index <- createDataPartition(df_data$fire_alarm, times = 1, p = 0.8, list = FALSE)

train_data <- df_data[train_index, ] %>% glimpse

# test data
test_data <- df_data[-train_index, ] %>% glimpse()


# modeling ----------------------------------------------------------------

# We don’t know what algorithms will perform well on this data before hand. We have to spot-check various different methods and see what looks good then double down on those methods.
# 
# ## Linear Algorithms:
# 1. Logistic Regression (LG),
# 
# 2. Linear Discriminate Analysis (LDA)
# 
# 3. Regularized Logistic Regression (GLMNET).
# 
# ## Non-Linear Algorithms:
# 1. k-Nearest Neighbors (KNN),
# 
# 2. Classification and Regression Trees (CART),
# 
# 3. Naive Bayes (NB)
# 
# 4. Support Vector Machines with Radial Basis Functions (SVM).
# 
# We have a good amount of data so we will use 10-fold cross validation with 3 repeats. This is a good standard test harness configuration. It is a binary classification problem. For simplicity, we will use Accuracy and Kappa metrics. We could have gone with the Area Under ROC Curve (AUC) and looked at the sensitivity and specificity to select the best algorithms.

# 10-fold cross validation with 3 repeats
trainControl <- trainControl(method="repeatedcv", number=10, repeats=3, verboseIter = TRUE)

metric <- "Accuracy"

# Build Model

# Bagged CART
set.seed(7)
fit.treebag <- train(fire_alarm~., data = train_data, method = "treebag", metric = metric,trControl = trainControl)

# RF
set.seed(7)
fit.rf <- train(fire_alarm~., data = train_data, method = "rf", metric = metric,trControl = trainControl)

# GBM - Stochastic Gradient Boosting
set.seed(7)
fit.gbm <- train(fire_alarm~., data = train_data, method = "gbm",metric = metric,trControl = trainControl, verbose = FALSE)

# C5.0
set.seed(7)
fit.c50 <- train(fire_alarm~., data = train_data, method = "C5.0", metric = metric,trControl = trainControl)

# LG - Logistic Regression
set.seed(7)
fit.glm <- train(fire_alarm~., data = train_data, method="glm",
                 metric=metric,trControl=trainControl)
# LDA - Linear Discriminate Analysis
set.seed(7)
fit.lda <- train(fire_alarm~., data = train_data, method="lda",
                 metric=metric,trControl=trainControl)

# GLMNET - Regularized Logistic Regression
set.seed(7)
fit.glmnet <- train(fire_alarm~., data = train_data, method="glmnet",
                    metric=metric,trControl=trainControl)

# KNN - k-Nearest Neighbors 
set.seed(7)
fit.knn <- train(fire_alarm~., data = train_data, method="knn",
                 metric=metric,trControl=trainControl)

# CART - Classification and Regression Trees (CART), 
set.seed(7)
fit.cart <- train(fire_alarm~., data = train_data, method="rpart",
                  metric=metric,trControl=trainControl)

# NB - Naive Bayes (NB) 
set.seed(7)
Grid = expand.grid(usekernel=TRUE,adjust=1,fL=c(0.2,0.5,0.8))
fit.nb <- train(fire_alarm~., data = train_data, method="nb",
                metric=metric,trControl=trainControl,
                tuneGrid=Grid)

# SVM - Support Vector Machines with Radial Basis Functions (SVM).
# set.seed(7)
# fit.svm <- train(fire_alarm~., data = train_data, method="svmRadial",
# metric=metric,trControl=trainControl)

# Comparing result
#After build the model compare model to find better accuracy

ensembleResults <- resamples(
  list(BAG = fit.treebag,
       RF = fit.rf,
       GBM = fit.gbm,
       C50 = fit.c50,
       LG = fit.glm,
       KNN = fit.knn,
       NB = fit.nb,
       # SVM = fit.svm,
       CART = fit.cart,
       GLMNET = fit.glmnet)
)
summary(ensembleResults)


# Plotting each model

dotplot(ensembleResults)

# based on running Random Forest is highest accuracy (99.99%), following by BAG (Bagged CART) (99.98%) and C5.0 (99.93%)

# Finalize Model
# Tree algorithms with higher accuracy will be selected for prediction: Random Forest, BAG and C5.0

# save model
saveRDS(fit.c50, here::here("finalModel_c50.rds"))
saveRDS(fit.rf, here::here("finalModel_rf.rds"))
saveRDS(fit.treebag, here::here("finalModel_treebag.rds"))


# testing model & accuracy ------------------------------------------------

# Decision Tree C5.0
model_c50 <- readRDS(here::here("finalModel_c50.rds"))

print(model_c50)

predict_c50 <- predict(model_c50, test_data)
summary(predict_c50)

# Confusion Matrix
cf_c50 <- confusionMatrix(predict_c50, test_data$fire_alarm)

cf_c50


# Decision Tree Random Forest
model_rf <- readRDS(here::here("finalModel_rf.rds"))

print(model_rf)
predict_rf <- predict(model_rf, test_data)
summary(predict_rf)

# Confusion Matrix
cf_rf <- confusionMatrix(predict_rf, test_data$fire_alarm)

cf_rf


# Classification and Regression Trees (CART)
model_treebag <- readRDS(here::here("finalModel_treebag.rds"))

print(model_treebag)

predict_treebag <- predict(model_treebag, test_data)
summary(predict_treebag)

# Confusion Matrix
cf_treebag <- confusionMatrix(predict_treebag, test_data$fire_alarm)

cf_treebag
