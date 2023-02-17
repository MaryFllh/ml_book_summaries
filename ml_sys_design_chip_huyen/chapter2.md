## Introduction to Machine Learning System Design
### Types of ML Tasks:
1. Regression: Outputs are continuous values
1. Classification: Outputs are different categories  
    1. Binary: There are only two categories to predict and a sample can only belong to one or the other  
    1. Multicalss: There are more than two categories but each sample can only belong to one. At least 100 examples for each class in needed for training.   
        1. Low cardinality: Number of classes is low  
        1. High cardinality: Number of classes is high, e.g. tens of thousands
    1. Multilabel: There are multiple categories and each sample can belong to more than one. There are two approaches to such problems:      
        1. Treat as multiclassification problem: For example the in a multiclass classification for articles, assume there are 4 classes of *art*, *entertainment*, *business* and *technology*. Similar to a multiclassification problem any article's label will be presented with a vector of length 4. The difference is that there can be multiple ones in this vector as an article can belong to both art and entertainment, i.e. [1, 1, 0, 0].
        1. Treat as binary classification: In the example above, you can train 4 different models. One predicting the likelihood of an article belonging to art or not, another predicting whether it belongs to technology or not, and so on.

### Conflicting Objectives
When training a machine learning model there can be multiple things you would want to maximise or minimise, e.g. in training a model to generate recommendations for a social network's news feed, you might want to recommend posts based on their quality rank, i.e. minimise the quality loss and also recommend posts with highest likelihood of engagement (minimise engagement loss). There can be cases where these two objectives are in conflict, posts where are of low quality but engage a lot. There are two approaches to this problem:   
1. Weight each loss and combine them into one: $\alpha$ quality_loss + $\beta$ engagement_loss   
The problem with the approach is that any change in user behaviour that will need tuning of $\alpha$ and $\beta$, will require retraining the model.
1.  Train different models, one for each objective and then combine their predictions by weighting their predictions, e.g. $\alpha$ quality_score + $\beta$ engagement_score. The advantage of this approach is that you don't need to retrain the model and can just change the weights given to the predictions. For example if the quality of the feeds increase but the engagement decreases, you would want to give $\beta$ a higher weight. In the previous approach you need to retrain the model. Here, you can just change the weights. Another benefit of this decoupling is easier maintainance as different models might require different retraining schedules.