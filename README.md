# KNN vs. LM-KNN for Iris Flower Classification

## Overview

This project explores the comparison between K-Nearest Neighbors (KNN) and Local Mean K-Nearest Neighbors (LM-KNN) in classifying iris flowers based on four characteristic factors: petal length, petal width, sepal length, and sepal width. Given the intricate relationships between these factors and the recorded data, an accurate method is essential for precise classification of iris species.

## Methodology

### K-Nearest Neighbors (KNN)
KNN is a classification method that predicts the class of new data based on the class of its closest neighboring data. This method relies on the concept of distance between data points and uses a majority vote system to determine the class of new data. KNN is highly effective for datasets with closely related characteristics.

### Local Mean K-Nearest Neighbors (LM-KNN)
LM-KNN, like KNN, is also based on the concept of distance between data points. However, in LM-KNN, the class for new data is chosen by calculating the closest distance to the local average of each data class. Unlike KNN, LM-KNN does not rely on a majority vote system for classification.

## Experiment and Data
This research will develop software to classify the iris dataset into 3 classes. The dataset is obtained from the Weka platform. The software testing uses 2 datasets: 
- **Iris**: which has 4 attributes.
- **Iris2d**: which has 2 attributes.

The data details are presented in Table 2 and Table 3.

![Screenshot 2024-06-28 220321](https://github.com/jakikbae/Performance-Comparison-of-KNN-and-LMKNN-Methods-for-Iris-Classification-with-Variation-of-Data/assets/88555527/b703b03e-6e19-4d42-b063-0da508fbb4b3)
![Screenshot 2024-06-28 220335](https://github.com/jakikbae/Performance-Comparison-of-KNN-and-LMKNN-Methods-for-Iris-Classification-with-Variation-of-Data/assets/88555527/435f9141-bfc5-4897-90ae-0b7123fd1249)


### Data Analysis and Feature Selection
- Determine parameters such as the K value.
- Implement distance calculations between test data and training data.
- Sort the data from smallest to largest distance.

### Classification
The class of data will be determined from a number of K nearest neighbors based on the results of the distance calculations. The results of the classification will be modeled and analyzed using the Confusion Matrix and K-Fold Cross Validation methods.

## Goals
The primary goals of this project are:
- To compare the accuracy and stability of KNN and LM-KNN methods.
- To assess the effect of dimensionality on the classification results.
- To identify the most effective method for classifying iris flowers based on the given attributes.

## Evaluation
The performance of the KNN and LM-KNN methods will be evaluated using:
- **Confusion Matrix**: To visualize the performance of the classification algorithm.
- **K-Fold Cross Validation**: To ensure the robustness and reliability of the results.

## Conclusion
From the results of this research, several conclusions can be drawn:
- The spread of distance between adjacent data can cause prediction errors that affect accuracy fluctuations.
- Software for classifying iris data using the KNN and LM-KNN methods has been successfully built.
- The use of different k values and data does not significantly affect performance. The highest result was found with k = 5 in the KNN method on the Iris dataset, achieving 97% accuracy, 98% precision, 97% recall, and 97% F1 score. Data with 3 attributes also reached a high accuracy of 97%.
- The KNN algorithm produced the highest accuracy value of 97%, while the LM-KNN algorithm achieved a highest accuracy of 96%.

## Suggestions
Based on the analysis and experiments, suggestions will be provided to improve the performance of the classification methods for future research and applications.
