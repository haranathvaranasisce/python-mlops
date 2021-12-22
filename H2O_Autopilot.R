library(h2o)
library(zoo)
library(xts)
h2o.init()

data=read.csv("AllDataWithDummies.csv")

#data$TARGET<-data$TARGET

#mdfh2o <- h2o.importFile("keepdf_lasso.csv") 
mdfh2o <- as.h2o(data)
h2odf <- h2o.splitFrame(data=mdfh2o, ratios=0.75)


 h2otrain <- h2odf[[1]]

# Create a testing set from the 2nd dataset in the split
h2otest <- h2odf[[2]]
###

# Identify predictors and response
y <- "TARGET"
x <- setdiff(names(mdfh2o), c("date","TARGET","CHANGE","OUTLIER","X"))
# Run AutoML for 20 base models (limited to 1 hour max runtime by default)
aml <- h2o.automl(x =x,
                  y = y,
                  training_frame = h2otrain,
                  max_models = 20,
                  seed = 1)

lb <- aml@leaderboard
print(lb, n = nrow(lb))  # Print all rows instead of default (6 rows)
pred <- h2o.predict(aml@leader, h2otest)
myaccuracy(h2otest[,"TARGET"],pred)

h2orf<-h2o.randomForest(y = y, x = x, training_frame = h2otrain)

pred1 <- h2o.predict(h2orf, h2otest)

myaccuracy(h2otest[,"TARGET"],pred1)
#View(aml@leader@model$variable_importances[order(-aml@leader@model$variable_importances$percentage),])

#data_test=read.csv("keepdf_test.csv")
#data_test$TARGET<-data_test$TARGET
#mdfh2o_test <- as.h2o(data_test)
#pred_test <- h2o.predict(aml@leader, mdfh2o_test)

#pred_test1 <- h2o.predict(h2orf, mdfh2o_test)
#myaccuracy(mdfh2o_test[,"TARGET"],pred_test1)

#View(h2orf@model$variable_importances[order(-h2orf@model$variable_importances$percentage),]
#View(as.data.frame(aml@leaderboard))
