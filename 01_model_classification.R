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