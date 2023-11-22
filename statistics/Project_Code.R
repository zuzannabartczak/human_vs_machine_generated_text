rm(list = ls())

library(ggplot2)
library(dplyr)
library(nortest)
library(jsonlite)
library(tidyverse)
library(pROC)
library(reshape2)


#Load data

train_data <- read.csv("data_csv/subtaskA_train_monolingual.csv", header = TRUE)

sgd_lc <- read.csv("SGD_outputs/learning_curve.csv", header = TRUE)

sgd_report_test <- read.csv("SGD_outputs/classification_report.csv", header = TRUE)
rnn_report_test <- read.csv("RNN_outputs/classification_report.csv", header = TRUE)
rownames(sgd_report_test) <- c("0", "1", "Accuracy", "Macro Avg", "Weighted Avg")
rownames(rnn_report_test) <- c("0", "1", "Accuracy", "Macro Avg", "Weighted Avg")

sgd_weights <- read.csv("SGD_outputs/weights.csv", header = TRUE)
rnn_weights <- read.csv("RNN_outputs/weights.csv", header = TRUE)

sgd_top <- read.csv("SGD_outputs/top_bottom_words.csv", header = TRUE)
rnn_top <- read.csv("RNN_outputs/top_bottom_words.csv", header = TRUE)

sgd_roc <- read.csv("SGD_outputs/ROC.csv", header = TRUE)
rnn_roc <- read.csv("RNN_outputs/ROC.csv", header = TRUE)

sgd_report_dev <- read.csv("SGD_outputs/classification_report2.csv", header = TRUE)
rnn_report_dev <- read.csv("RNN_outputs/classification_report2.csv", header = TRUE)
rownames(sgd_report_dev) <- c("0", "1", "Accuracy", "Macro Avg", "Weighted Avg")
rownames(rnn_report_dev) <- c("0", "1", "Accuracy", "Macro Avg", "Weighted Avg")



# Distribution of human vs. machine-generated texts

# Remove rows with NA in 'model' or 'label' column
train_data <- train_data %>% drop_na(model, label)

ggplot(train_data, aes(x = model, fill = model)) +
    geom_bar(position = "dodge") +
    ggtitle("Distribution of Human vs. Machine-Generated Texts")


ggplot(train_data, aes(x = model, fill = source)) +
    geom_bar(position = "dodge") +
    ggtitle("Distribution of domains")


# Calculate text length
train_data$length <- str_length(train_data$text)

# Define the percentiles to remove from each side
lower_percentile <- 0.10  # Remove bottom 10%
upper_percentile <- 0.90  # Remove top 10%

# Calculate the corresponding text length for each percentile
lower_limit <- quantile(train_data$length, lower_percentile)
upper_limit <- quantile(train_data$length, upper_percentile)

# Filter out texts outside the desired range
length_filtered <- train_data %>% filter(length >= lower_limit & length <= upper_limit)

# Plot the distribution of text length
ggplot(train_data, aes(x = length)) +
    geom_histogram(binwidth = 5, fill = "dodgerblue2", color = "dodgerblue2") +
    labs(title = "Distribution of Text Length", x = "Text Length", y = "Frequency")


# Test for normality using the Anderson-Darling test
ad_test <- ad.test(as.numeric(train_data$label))
cat("Anderson-Darling test statistic:", ad_test$statistic, "\n")
cat("p-value:", ad_test$p.value, "\n")
if (ad_test$p.value > 0.05) {
    cat("The data is normally distributed.\n")
} else {
    cat("The data is not normally distributed.\n")
}


# Boxplot of predicted probabilities for RNN model
ggplot(rnn_roc, aes(x = actual, y = predicted, fill = actual)) +
    geom_boxplot() +
    labs(title = "RNN Model - Predicted Probabilities for Class 1")


# Boxplot of predicted probabilities for both classes (0 and 1) in SGD model
ggplot(sgd_roc, aes(x = actual, y = prob_0, fill = actual)) +
    geom_boxplot() +
    labs(title = "SGD Model - Predicted Probabilities for Class 0")

ggplot(sgd_roc, aes(x = actual, y = prob_1, fill = actual)) +
    geom_boxplot() +
    labs(title = "SGD Model - Predicted Probabilities for Class 1")

lower_percentile <- 0.15
upper_percentile <- 0.85

lower_limit <- quantile(sgd_weights$weight, lower_percentile)
upper_limit <- quantile(sgd_weights$weight, upper_percentile)

sgd_weights_filtered <- sgd_weights %>% filter(weight >= lower_limit & weight <= upper_limit)

intervals <- seq(-0.001114333, 0.000313608, length.out = 6)
intervals <- round(intervals, digits = 9)
print(intervals)

# Plot the density plot
ggplot(sgd_weights_filtered, aes(x = weight)) +
    geom_density(fill = "darksalmon") +
    labs(x = "Weights", y = "Density", title = "SGD - Density Plot of Weights") +
    scale_x_continuous(breaks = intervals[1:6], labels = c('-0.001114333', '-0.000828745', '-0.000543157', '-0.000257568', '0.000028020', '0.000313608'))

ggplot(sgd_top, aes(x = reorder(word, weight), y = weight)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    labs(x = "Words", y = "Weights", title = "SGD - Top and Bottom Words by Weight")

# Plot the density plot
ggplot(rnn_weights, aes(x = weight)) +
    geom_density(fill = "olivedrab") +
    labs(x = "Weights", y = "Density", title = "RNN - Density Plot of Weights")

ggplot(rnn_top, aes(x = reorder(word, weight), y = weight)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    labs(x = "Words", y = "Weights", title = "RNN - Top and Bottom Words by Weight")


sgd_sample <- sgd_weights[sample(nrow(sgd_weights), 10000), ]

# Plot the scatter plot
ggplot() +
    geom_point(data = sgd_sample, aes(x = word, y = weight, color = "SGD")) +
    geom_point(data = rnn_weights, aes(x = word, y = weight, color = "RNN")) +
    labs(title = "Comparison of Word Weights between SGD and RNN Models", x = "Words", y = "Weights") +
    scale_color_manual(values = c("SGD" = "darkgreen", "RNN" = "purple"))


rnn_cm <- data.frame(
    actual = c("0", "0", "1", "1"),
    predicted = c("0", "1", "0", "1"),
    count = c(7376, 5120, 5546, 5910)
)

# Plot the confusion matrix
ggplot(rnn_cm, aes(x = actual, y = predicted, fill = count)) +
    geom_tile() +
    geom_text(aes(label = count)) +
    scale_fill_gradient(low = "lightgrey", high = "purple") +
    labs(title = "RNN Confusion Matrix", x = "Actual", y = "Predicted", fill = "Count") +
    theme_minimal()


sgd_cm <- data.frame(
    actual = c("0", "0", "1", "1"),
    predicted = c("0", "1", "0", "1"),
    count = c(11247,1249, 5605,5851)
)

# Plot the confusion matrix
ggplot(sgd_cm, aes(x = actual, y = predicted, fill = count)) +
    geom_tile() +
    geom_text(aes(label = count)) +
    scale_fill_gradient(low = "lightgrey", high = "darkgreen") +
    labs(title = "SGD Confusion Matrix", x = "Actual", y = "Predicted", fill = "Count") +
    theme_minimal()

# Plot the learning curve
ggplot(sgd_lc, aes(train_sizes)) +
    geom_line(aes(y = train_scores_mean), color = "hotpink3") +
    geom_line(aes(y = test_scores_mean), color = "darkslateblue") +
    geom_ribbon(aes(ymin = train_scores_mean - train_scores_std, ymax = train_scores_mean + train_scores_std), alpha = 0.1, fill = "hotpink3") +
    geom_ribbon(aes(ymin = test_scores_mean - test_scores_std, ymax = test_scores_mean + test_scores_std), alpha = 0.1, fill = "darkslateblue") +
    labs(title = "SGD Learning Curve", x = "Training Size", y = "Score") +
    theme_minimal()

# Plot the ROC curve
roc_rnn <- roc(rnn_roc$actual, rnn_roc$predicted)

plot(roc_rnn, main="RNN ROC Curve")
abline(a=0, b=1, lty=2, col="gray") 

auc(roc_rnn)



rnn_report_test$vartag <- row.names(rnn_report_test)
rnn_long_test <- melt(rnn_report_test, "vartag")

rnn_long_test <- rnn_long_test %>%
    filter(!(variable == "support" | vartag %in% c("Accuracy", "Macro Avg", "Weighted Avg")))

sgd_report_test$vartag <- row.names(sgd_report_test)
sgd_long_test <- melt(sgd_report_test, "vartag")

sgd_long_test <- sgd_long_test %>%
    filter(!(variable == "support" | vartag %in% c("Accuracy", "Macro Avg", "Weighted Avg")))


max_value <- max(max(sgd_long_test$value), max(rnn_long_test$value))

ggplot(sgd_long_test, aes(x = variable, y = value, fill = variable)) +
    geom_bar(stat = "identity", position = "dodge") +
    facet_wrap(~ vartag, ncol = 2) +
    labs(title = "SGD_test - Precision, Recall, and F1-Score 
         for Classes 0 and 1") +
    coord_cartesian(ylim = c(0, max_value))

ggplot(rnn_long_test, aes(x = variable, y = value, fill = variable)) +
    geom_bar(stat = "identity", position = "dodge") +
    facet_wrap(~ vartag, ncol = 2) +
    labs(title = "RNN_test - Precision, Recall, and F1-Score 
         for Classes 0 and 1") +
    coord_cartesian(ylim = c(0, max_value))

rnn_report_dev$vartag <- row.names(rnn_report_dev)
rnn_long_dev <- melt(rnn_report_dev, "vartag")

rnn_long_dev <- rnn_long_dev %>%
    filter(!(variable == "support" | vartag %in% c("Accuracy", "Macro Avg", "Weighted Avg")))

sgd_report_dev$vartag <- row.names(sgd_report_dev)
sgd_long_dev <- melt(sgd_report_dev, "vartag")

sgd_long_dev <- sgd_long_dev %>%
    filter(!(variable == "support" | vartag %in% c("Accuracy", "Macro Avg", "Weighted Avg")))


max_value <- max(max(sgd_long_dev$value), max(rnn_long_dev$value))

ggplot(sgd_long_dev, aes(x = variable, y = value, fill = variable)) +
    geom_bar(stat = "identity", position = "dodge") +
    facet_wrap(~ vartag, ncol = 2) +
    labs(title = "SGD_dev - Precision, Recall, and F1-Score 
         for Classes 0 and 1") +
    coord_cartesian(ylim = c(0, max_value))

ggplot(rnn_long_dev, aes(x = variable, y = value, fill = variable)) +
    geom_bar(stat = "identity", position = "dodge") +
    facet_wrap(~ vartag, ncol = 2) +
    labs(title = "RNN_dev - Precision, Recall, and F1-Score 
         for Classes 0 and 1") +
    coord_cartesian(ylim = c(0, max_value))

