# Supervised Learning with Text
rm(list = ls())
setwd("C:/Users/dhumm/Desktop/Text_as_Data/Corpus_work/Reboot2")
# load in packages
library(quanteda)
require(quanteda)
# devtools::install_github("matthewjdenny/SpeedReader")
library(SpeedReader)

# Install som packages we are going to use:
#install.packages(
#  c("xgboost","glmnet","caret","ROCR","pROC"),
#  dependencies = TRUE)

library(xgboost)
library(caret)
library(ROCR)
library(glmnet)
library(pROC)


df <- read.csv('Processed_Corpus.csv')
df = subset(df, select = -c(All_Text) )


colnames(df)[1] <- gsub('^...','',colnames(df)[1])

df$Processed_text <- as.character(df$Processed_text)

Encoding(df$Processed_text) <- "UTF-8"

df$Processed_text <- gsub('\\[', '', df$Processed_text)
df$Processed_text <- gsub('\\]', '', df$Processed_text)
df$Processed_text <- gsub('\\n', '', df$Processed_text)
df$Processed_text <- gsub('â???"', '', df$Processed_text)
df$Processed_text <- gsub('Â', '', df$Processed_text)
df$Processed_text <- gsub('amp;', '', df$Processed_text)

df$Processed_text <- gsub("'", '', df$Processed_text)


## Take out covid-19 and coronavirus references
df$Processed_text <- gsub('coronavirus', '', df$Processed_text)
df$Processed_text <- gsub('corona', '', df$Processed_text)
df$Processed_text <- gsub('covid-19', '', df$Processed_text)
df$Processed_text <- gsub('covid19', '', df$Processed_text)
df$Processed_text <- gsub('covid', '', df$Processed_text)
df$Processed_text <- gsub('recent_year_number-ncov', '', df$Processed_text)

## Second round of covid references
df$Processed_text <- gsub('please', '', df$Processed_text)
df$Processed_text <- gsub('visit', '', df$Processed_text)
df$Processed_text <- gsub('national', '', df$Processed_text)
df$Processed_text <- gsub('emergency', '', df$Processed_text)
df$Processed_text <- gsub('outbreak', '', df$Processed_text)
df$Processed_text <- gsub('cares', '', df$Processed_text)
df$Processed_text <- gsub('cdc.gov', '', df$Processed_text)
df$Processed_text <- gsub('index.html', '', df$Processed_text)

# note that we need to specify a docid field and a text field
my_corp <- corpus(df,
                  docid_field = "ID_num",
                  text_field = "Processed_text")

summary(my_corp)

# Tokenize and apply ngrams
toks = tokens(my_corp,remove_punct = TRUE,remove_numbers = TRUE)

toks_ngram <- tokens_ngrams(toks, n = 1:4)


# process our data:
dtm <- dfm(toks_ngram,
           remove = stopwords("english"))

# look at vocabulary size
dtm

# now lets trim the vocabulary to make things easier to work with:
dtm <- dfm_trim(dtm,
                min_termfreq = 3,
                min_docfreq = 3,
                max_docfreq = 363)

# look at vocabulary size
dtm

# Create a tf_idf out of our dtm
tf_idf <- dfm_tfidf(dtm)

tf_idf

# pull out the document level covariates:
features <- docvars(tf_idf)

################# GLMNET #################
# lets start by training a supervised classifier for a binary classification
# problem using a lasso (regularized) logistic regression model.
# See: https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html

### Post_Model_E
# partition our data into train and test sets:
set.seed(1234)
trainIndex <- createDataPartition(features$Post_Model_E,
                                  p = 0.8,
                                  list = FALSE,
                                  times = 1)

# pull out the first column as a vector:
trainIndex <- trainIndex[,1]

train <- dtm[trainIndex, ]
test <- dtm[-trainIndex, ]

# Create separate vectors of our outcome variable for both our train and test sets
# We'll use these to train and test our model later
train.label  <- features$Post_Model_E[trainIndex]
test.label   <- features$Post_Model_E[-trainIndex]

# train our lasso
cvfit = cv.glmnet(x = train,
                  y = train.label,
                  family = "binomial",
                  type.measure = "class")

pdf(file = "Optimal_Lasso_Penalty_Model_E3.pdf",
    width = 10,
    height = 5)
plot(cvfit)
dev.off()
plot(cvfit)


# lets take a look at the coefficients:
head(coef(cvfit, s = "lambda.min"),n = 50)

# make predictions
pred <- predict(
  cvfit,
  newx = test,
  s = "lambda.min",
  type = "response")

# select a threshold and generate predcited labels:
pred_vals <- ifelse(pred >= 0.5, 1, 0)

# Create the confusion matrix
confusionMatrix(table(pred_vals, test.label),positive="1")


# Use ROCR package to plot ROC Curve
lasso.pred <- prediction(pred, test.label)
lasso.perf <- performance(lasso.pred, "tpr", "fpr")

pdf(file = "LASSO_ROC_Model_E3.pdf",
    width = 6,
    height = 6)
plot(lasso.perf,
     avg = "threshold",
     colorize = TRUE,
     lwd = 1,
     main = "Lasso ROC Curve w/ Thresholds: Model E3",
     print.cutoffs.at = c(.9,.8,.7,.6,.5,.4,.3,.2,.1),
     text.adj = c(-0.5, 0.5),
     text.cex = 0.5)
grid(col = "lightgray")
axis(1, at = seq(0, 1, by = 0.1))
axis(2, at = seq(0, 1, by = 0.1))
abline(v = c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
abline(h = c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
lines(x = c(0, 1), y = c(0, 1), col="black", lty="dotted")
dev.off()

plot(lasso.perf,
     avg = "threshold",
     colorize = TRUE,
     lwd = 1,
     main = "Lasso ROC Curve w/ Thresholds: Model E3",
     print.cutoffs.at = c(.9,.8,.7,.6,.5,.4,.3,.2,.1),
     text.adj = c(-0.5, 0.5),
     text.cex = 0.5)
grid(col = "lightgray")
axis(1, at = seq(0, 1, by = 0.1))
axis(2, at = seq(0, 1, by = 0.1))
abline(v = c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
abline(h = c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
lines(x = c(0, 1), y = c(0, 1), col="black", lty="dotted")

# we can also get the AUC for this predictor:
auc.perf = performance(lasso.pred,
                       measure = "auc")
auc.perf@y.values[[1]]

# and look at accuracy by threshold
acc.perf = performance(lasso.pred, measure = "acc")
#plot(acc.perf)

# we can also calculate the optimal accuracy and its associated threshold:
ind = which.max( slot(acc.perf, "y.values")[[1]] )
acc = slot(acc.perf, "y.values")[[1]][ind]
cutoff = slot(acc.perf, "x.values")[[1]][ind]
print(c(accuracy= acc, cutoff = cutoff))