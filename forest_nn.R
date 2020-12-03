################ NN algo on forestfires dataset #########################
forest_data <- forestfires
dim(forest_data)
names(forest_data)
head(forest_data)
forest_data <- forest_data[c(-1,-2)]
str(forest_data)

forest_data$size_category <- as.numeric(as.factor(forest_data$size_category))

library(neuralnet)
library(caret)

normalize <- function(x)
            {
                return((x-min(x)/max(x)-min(x)))
}

forestfires_n <- as.data.frame(lapply(forest_data,normalize))     ## normalizing data 
head(forestfires_n)
forestfires_n$FFMC <- round(forestfires_n$FFMC, digits = 2)       ## rounding data upto two decimal points 
forestfires_n$DMC <- round(forestfires_n$DMC, digits = 2)
forestfires_n$DC <- round(forestfires_n$DC, digits = 2)
forestfires_n$temp <- round(forestfires_n$temp, digits = 2)
forestfires_n$wind <- round(forestfires_n$wind, digits = 2)

partition <- createDataPartition(forestfires_n$size_category,p=.75,list = F)  ## partitioning data 
train_data <- forestfires_n[partition,]
test_data <- forestfires_n[-partition,]

sim_model <- neuralnet(train_data$size_category~.,data = train_data)
plot(sim_model)
sim_model
dim(train_data)
sim_pred <- compute(sim_model,train_data[-29])
sim_pred
sim_result <- sim_pred$net.result
cor(sim_result,train_data$size_category)

sim_pred_test <- compute(sim_model,test_data[-29])
sim_pred_test
sim_result_test <- sim_pred_test$net.result
cor(sim_result_test,test_data$size_category)

##### adding more hidden layer and neurons ############
adv_model <- neuralnet(train_data$size_category~.,data = train_data,hidden = c(2,2))

adv_pred <- compute(adv_model,train_data[-29])
adv_pred
sim_result <- adv_pred$net.result
cor(sim_result,train_data$size_category)

adv_pred_test <- compute(adv_model,test_data[-29])
adv_pred_test
sim_result_test <- adv_pred_test$net.result
cor(sim_result_test,test_data$size_category)

### adding sum etra info to model ################3
hibrid_model <- neuralnet(train_data$size_category~.,data = train_data,hidden = c(2,2),algorithm = 'backprop'
                          , learningrate = 0.00001,linear.output=F,stepmax=1e+08,act.fct = 'tanh')

hibrid_pred <- compute(hibrid_model,train_data[-29])
cor(hibrid_pred$net.result,train_data$size_category)

hibrid_pred_test <- compute(hibrid_model,test_data[-29])
cor(hibrid_pred_test$net.result,test_data$size_category)


