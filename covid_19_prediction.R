library(prophet)
library(lubridate)
#library(dplyr)
library(ggplot2)

file <- read.csv("total_cases.csv", colClasses = "character") # read data

data <- data.frame(as.Date(file$date), as.numeric(file$India), as.numeric(file$United.States)) # make a new dataframe and change the datatypes
names(data) <- c("Date", "India", "US") # rename dataframe columns

# Prediction of Covid 19 Cases in US

ggplot(data, aes(Date, India)) + geom_point(colour = "blue") #plot Covid Cases in India
 
ds <- data$Date
y <- data$India
df <- data.frame(ds, y)

model <- prophet(df) 
future_cases <- make_future_dataframe(model, periods = 30) 
prediction <- predict(model, future_cases)

plot(model, prediction, main= "COVID-19 Prediction", sub="India", xlab="Day", ylab="Total Cases")
 
predicted_IN <- prediction$yhat[1:245]
actual_IN <-model$history$y

dev.new()
plot(actual_IN, predicted_IN, main= "Model Performance", sub = "India", xlab="Actual", ylab="Predicted")

# Prediction of Covid 19 Cases in US

dev.new()
ggplot(data, aes(Date, US)) + geom_point(colour = "red")

ds <- data$Date
y <- data$US
df <- data.frame(ds, y)

model <- prophet(df)
future_cases <- make_future_dataframe(model, periods = 30)
prediction <- predict(model, future_cases)

plot(model, prediction, main= "COVID-19 Prediction", sub="US", xlab="Day", ylab="Total Cases")
 
predicted_US <- prediction$yhat[1:246]
actual_US <-model$history$y

dev.new()
plot(actual_US, predicted_US, main= "Model Performance", sub = "US", xlab="Actual", ylab="Predicted")