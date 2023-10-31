library(ggplot2)
library(dplyr)
library(nortest)
library(jsonlite)
library(tidyverse)


#Load data

train_data <- read.csv("data_csv/subtaskA_train_monolingual.csv", header = TRUE)
dev_data <- read.csv("data_csv/subtaskA_dev_monolingual.csv", header = TRUE)

sgd_lc <- read.csv("SGD_outputs/learning_curve.csv", header = TRUE)

sgd_cm <- read.csv("SGD_outputs/confusion_matrix.csv", header = TRUE)
rnn_cm <- read.csv("RNN_outputs/confusion_matrix.csv", header = TRUE)

sgd_report1 <- read.csv("SGD_outputs/classification_report.csv", header = TRUE)
rnn_report1 <- read.csv("RNN_outputs/classification_report.csv", header = TRUE)

sgd_weights <- read.csv("SGD_outputs/weights.csv", header = TRUE)
rnn_weights <- read.csv("RNN_outputs/weights.csv", header = TRUE)

sgd_top <- read.csv("SGD_outputs/top_bottom_words.csv", header = TRUE)
rnn_top <- read.csv("RNN_outputs/top_bottom_words.csv", header = TRUE)

sgd_roc <- read.csv("SGD_outputs/ROC.csv", header = TRUE)
rnn_roc <- read.csv("RNN_outputs/ROC.csv", header = TRUE)

sgd_report2 <- read.csv("SGD_outputs/classification_report2.csv", header = TRUE)
rnn_report2 <- read.csv("RNN_outputs/classification_report2.csv", header = TRUE)


lines <- readLines("SGD_outputs/dev_predictions.jsonl")
lines <- lapply(lines, fromJSON)
sgd_pred <- bind_rows(lines)

lines2 <- readLines("RNN_outputs/dev_predictions.jsonl")
lines2 <- lapply(lines2, fromJSON)
rnn_pred <- bind_rows(lines2)



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
ggplot(length_filtered, aes(x = length)) +
    geom_histogram(binwidth = 5, fill = "lightblue", color = "lightblue") +
    labs(title = "Distribution of Text Length", x = "Text Length", y = "Frequency")



labels <- train_data$label
labels <- as.numeric(labels)

# Test for normality using the Anderson-Darling test
ad_test <- ad.test(labels)

# Print the test statistic and p-value
cat("Anderson-Darling test statistic:", ad_test$statistic, "\n")
cat("p-value:", ad_test$p.value, "\n")

# Interpret the results
if (ad_test$p.value > 0.05) {
    cat("The data is normally distributed.\n")
} else {
    cat("The data is not normally distributed.\n")
}


# Print the mean and standard deviation for the SGD model
cat("SGD model:\n")
cat("Mean:", sgd_lc$train_scores_mean, "\n")
cat("Standard deviation:", sgd_lc$train_scores_std, "\n")

# Print the mean and standard deviation for the RNN model
cat("RNN model:\n")
cat("Mean:", rnn_mean, "\n")
cat("Standard deviation:", rnn_std, "\n")



# Test for statistical significance using the Mann-Whitney U test (between training and test or sgd and rnn??)
wilcox_test <- wilcox.test(rnn_data, sgd_data)

# Print the test statistic and p-value
cat("Mann-Whitney U test statistic:", wilcox_test$statistic, "\n")
cat("p-value:", wilcox_test$p.value, "\n")

# Interpret the results
if (wilcox_test$p.value > 0.05) {
    cat("There is no significant difference between the RNN and SGD models.\n")
} else {
    cat("There is a significant difference between the RNN and SGD models.\n")
}


names(sgd_lc) <- c("TrainSize", "TrainScore", "TestScore", "TrainStdDev", "TestStdDev")

# Plot the learning curve
ggplot(sgd_lc, aes(TrainSize)) +
    geom_line(aes(y = TrainScore), color = "darkgreen") +
    geom_line(aes(y = TestScore), color = "purple") +
    geom_ribbon(aes(ymin = TrainScore - TrainStdDev, ymax = TrainScore + TrainStdDev), alpha = 0.1, fill = "darkgreen") +
    geom_ribbon(aes(ymin = TestScore - TestStdDev, ymax = TestScore + TestStdDev), alpha = 0.1, fill = "purple") +
    labs(title = "Learning Curve", x = "Training Size", y = "Score") +
    scale_x_continuous(expand = c(0, 0)) +
    scale_y_continuous(expand = c(0, 0)) +
    theme_minimal()



# Plot weights
ggplot(sgd_weights, aes(x = word, y = weight)) +
    geom_bar(stat = 'identity') +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    labs(title = "Feature Weights", x = "Feature", y = "Weight")


# Plot the feature importance
ggplot(df, aes(x = reorder(Feature, Importance), y = Importance)) +
    geom_bar(stat = 'identity') +
    coord_flip() +
    labs(title = "Feature Importance", x = "Feature", y = "Importance")

# Define the confusion matrix
cm1 <- matrix(c(12198, 298, 7590, 3866), nrow = 2, byrow = TRUE)

# Convert the confusion matrix to a data frame
cm_sgd <- data.frame(
    actual = rep(c("Positive", "Negative"), each = 2),
    predicted = rep(c("Positive", "Negative"), times = 2),
    count = as.vector(cm1)
)

# Plot the confusion matrix
ggplot(cm_sgd, aes(x = actual, y = predicted, fill = count)) +
    geom_tile() +
    geom_text(aes(label = count)) +
    scale_fill_gradient(low = "lightgrey", high = "purple") +
    labs(x = "Actual", y = "Predicted", fill = "Count") +
    theme_minimal()
