# UCI Wine Quality Dataset Analysis

## Table of Contents
1. [Introduction](#introduction)
2. [Exploratory Data Analysis](#eda)
3. [Wine Quality Modeling](#wine-quality-modeling)
4. [Model Results](#classifier-results)
5. [Using SMOTE for Class Imbalance](#tackling-imbalance-with-smote)
6. [Conclusion](#conclusion)

### Introduction

This wine quality dataset comes from UC Irvine's Machine Learning Repository (https://archive.ics.uci.edu/ml/datasets/wine+quality). The purpose of this analysis is to determine if we can accurately predict the quality of wine based off some of its chemical properties such as pH and sugar levels. There are two separate datasets - one for red wine and the other for white - but only the red wine dataset is analyzed here.

### EDA

The chemical properties of the wines are all continuous variables.
Quality ratings can range from 1 through 10, where lower values represent poorer quality, middle values represent normal quality, and higher values represent excellent quality. This dataset, however, only contains data with quality values from 3 through 8 and is unbalanced, with more data points in the normal quality value range as seen in Figure 1 below.

<img alt="Quality Rating Counts" src="images/quality_rating_counts.png">

Lucky for us, the dataset was cleaned up by UCI prior to posting in their repository so there are no null values present.

Next, I looked at the distribution of each of the predictor variables and found many have outliers, usually above the upper threshold to be considered an outlier. Again, since the dataset was cleaned up, I don't expect any errors with how the data in this set was collected. I looked at the quality values of the outliers for each chemical property and most had similar ratings. Therefore, points with outliers were kept in the dataset as they could contain important information in finding a relationship between wine properties and their qualities. The boxplots for each variable are in Figure 2 below.

<img alt="Boxplots of Each Feature" src="images/feature_boxplots.png">

Figure 3 is a heatmap illustrating the correlations between all the variables. One thing that caught my attention were the two columns, free sulfur dioxide and total sulfur dioxide, which have a relatively high correlation in the heatmap. I'm not a chemistry or wine expert but my hunch was that total SO2 might be a combination of free sulfur dioxide and something else. This thought was confirmed after doing some research so I removed the free sulfur dioxide column to remove redundancy in the features.

There are also other variables that appear to be correlated but I did not conduct further research so more professional input could be used here. Free and total sulfur dioxide seemed like an immediate dead giveaway that there could be some redundancy there.

<img alt="Feature Correlations" src="images/correlations.png">

I explored the relationship between each feature and quality using the plots in Figure 4. Another reason why I wasn't too concerned with correlation between the variables is there does not seem to be any clear linear relationships between any of the predictor variables and the response so I am not planning on using any regressors or models where performance would be impacted by correlation. The relationship between alcohol levels and quality seems to be the strongest, which can be seen in the heatmap and the scatterplot, but the correlation level is still only 0.48.

<img alt="Features vs. Quality" src="images/features_vs_quality.png">

### Wine Quality Modeling

As stated, I have decided not to use a logistic regressor since the relationships between each of the predictor variables and quality appear to be more complex than a linear one. For this analysis, I looked at how well a relationship could be modeled by using a Random Forest Classifier and a Gradient Boosting Classifier. Although these two classifiers are complex algorithms, we can still determine which features are most important in modeling wine quality, given that we even find a relatively strong relationship between predictors and the response.

The data is split in train and test sets. The train set is then cross-validated via GridSearch, which is used to find optimal parameters for each of the classifiers. Even though there is class imbalance, the train and cross validated sets are stratified to ensure there are training points with each quality value. The performance of the model is validated by testing it on the hold out test set.

Here I am going to evaluate the classifiers based on their accuracy scores since we care about all the correct and incorrect predictions they make. I briefly discuss the effect of class imbalance in the conclusion.

### Classifier Results

After finding the best parameters for the Random Forest classifier, the training accuracy is 68.64% and the accuracy calculated with the hold out test set is 68%. The training accuracy for the Gradient Boosting Classifier with the optimal parameters is 65.64% and the test accuracy is 65%.

The confusion matrices show that a majority of quality ratings it predicted correctly are numbers in the normal wine quality range, which makes sense with the imbalanced data. Therefore, the models are making accurate predictions for average qualities but not for the extreme values. Again, I briefly talk about tackling class imbalance and model performance in the conclusion.  

### Tackling Imbalance with SMOTE

We discovered there was a large class imbalance for extreme ratings so I wanted to see if we could improve our training by using Synthetic Minority Over-Sampling Technique (SMOTE) to create synthetic points that make up for the imbalance in the lower and higher rating classes.

When I originally did this analysis, my models were overfitting the training data. There was overfitting because I created the train-test split and applied SMOTE to the train set. However, this presented a problem for the GridSearchCV because it was cross validating on whole set that had synthetic points. As a result, information in the validation sets was leaked into the individual training sets and the average accuracy was ballooned compared to the accuracy on the hold out test set.

To solve this, I created a pipeline object to pass through GridSearch where the pipeline included the SMOTE and classifier objects. This change effectively addressed the overfitting problem I mentioned above; when GridSearchCV did the cross validations, SMOTE was only applied to the individual training sets and not on the hold out validation set so the mean accuracy score was correctly represented.

Even though I was able to get SMOTE to work with GridSearchCV, I was not able to improve accuracy. In fact, the scores were lower to the ones when SMOTE was not applied. There just isn't enough data with the extreme ratings. The accuracy scores with SMOTE were lower because the models started categorizing points as extreme ratings that should have been classified as the average qualities (as shown in the confusion matrices). The models probably did this because the synthetic data points created by the very few points we have with extreme ratings have similar predictive feature values as the points with average ratings so the models were having trouble distinguishing between the two.


### Conclusion

Unfortunately, there does not appear to be an accurate way to predict wine qualities based off their chemical properties. Because I was not able to reach an accuracy higher than 80%, I did not move forward with looking at feature importances. Of course here I only looked at two different classifiers so it might be worth trying some others to see if they perform better with this dataset.

Finally, I looked at some analyses done by others on this dataset and they tended to engineer and narrow the response variable to different quality levels such as 'bad', 'good', 'excellent' to handle the class imbalance so that is something to consider. I was curious to see what would happen with SMOTE.
