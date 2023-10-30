library(ggplot2)
library(dplyr)

taskAmono <- read.csv("data_csv/subtaskA_train_monolingual.csv", header = TRUE)

# BEGIN: eda_visualizations

# Distribution of human vs. machine-generated texts
ggplot(taskAmono, aes(x = label)) +
    geom_bar() +
    ggtitle("Distribution of Human vs. Machine-Generated Texts in Monolingual Dataset")

ggplot(taskAmulti, aes(x = label)) +
    geom_bar() +
    ggtitle("Distribution of Human vs. Machine-Generated Texts in Multilingual Dataset")

# Distribution across different languages and domains
ggplot(taskB, aes(x = model, fill = label)) +
    geom_bar(position = "dodge") +
    ggtitle("Distribution of Human vs. Machine-Generated Texts Across Languages in Subtask B")


# Create a data frame with the weights
weights_df <- read.csv("weights.csv", header = TRUE)

# Plot the weights using ggplot2
ggplot(weights_df, aes(x = 1:length(weights), y = weights)) +
    geom_line() +
    xlab("Value") +
    ylab("Weights") +
    theme_minimal()


# Load the learning curve data
learning_curve <- read.csv("learning_curve.csv")

# Plot the learning curve
ggplot(learning_curve, aes(x = train_sizes)) +
    geom_line(aes(y = train_scores_mean), color = "blue", size = 1) +
    geom_line(aes(y = test_scores_mean), color = "red", size = 1) +
    geom_ribbon(aes(ymin = train_scores_mean - train_scores_std / 2,
                    ymax = train_scores_mean + train_scores_std / 2),
                fill = "blue", alpha = 0.2) +
    geom_ribbon(aes(ymin = test_scores_mean - test_scores_std / 2,
                    ymax = test_scores_mean + test_scores_std / 2),
                fill = "red", alpha = 0.2) +
    labs(x = "Training examples", y = "Score", color = "Legend") +
    scale_color_manual(values = c("blue", "red"), labels = c("Training score", "Cross-validation score")) +
    theme_minimal()

# Define the confusion matrix
confusion_matrix <- matrix(c(12198, 298, 7590, 3866), nrow = 2, byrow = TRUE)

# Convert the confusion matrix to a data frame
confusion_matrix_df <- data.frame(
    actual = rep(c("Positive", "Negative"), each = 2),
    predicted = rep(c("Positive", "Negative"), times = 2),
    count = as.vector(confusion_matrix)
)

# Plot the confusion matrix
ggplot(confusion_matrix_df, aes(x = actual, y = predicted, fill = count)) +
    geom_tile() +
    geom_text(aes(label = count)) +
    scale_fill_gradient(low = "white", high = "blue") +
    labs(x = "Actual", y = "Predicted", fill = "Count") +
    theme_minimal()




# Read the learning curve data from the csv file
learning_curve_df <- read.csv("RNN_outputs/learning_curve.csv")

# Plot the learning curve
ggplot(learning_curve_df, aes(x = Training.Sizes)) +
    geom_line(aes(y = Training.Scores.Mean), color = "red") +
    geom_line(aes(y = Validation.Scores.Mean), color = "green") +
    geom_ribbon(aes(ymin = Training.Scores.Mean - Training.Scores.Std, ymax = Training.Scores.Mean + Training.Scores.Std), alpha = 0.1, fill = "red") +
    geom_ribbon(aes(ymin = Validation.Scores.Mean - Validation.Scores.Std, ymax = Validation.Scores.Mean + Validation.Scores.Std), alpha = 0.1, fill = "green") +
    labs(title = "Learning Curve", x = "Training Examples", y = "Score") +
    theme_minimal()