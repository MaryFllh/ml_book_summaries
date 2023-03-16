# Feature Engineering
## Common Feature Engineering Operations
### **Handling Missing Values**
Consider the problem of predicting whether someone is going to buy a house in the next 12 months with some sample data seen below. 
<center>
<img src="images/house_purchase.jpg" width="50%" alt="house_purchase" title="house_purchase">
</center>

There are three types of missing values:     
1. **Missing not at random (MNAR)**: The true value itself is the reason for it being missing. For example in the table above, some of the income values are missing. It might be turn out after investigating that people with higher income were not comfortable sharing that information. In other words, the value is missing because of the value itself.
1. **Missing at random (MAR)**: In this case the value is not missing because of the value itself but because of the value of another observed variable. In the table above, the age of people of gender A is missing. This might be because people of this gender (another observed variable) are not comfortable sharing their age. 
1. **Missing completely at random (MCAR)**: This is when there is no pattern in when the value is missing. For example, the Job values in the above examples may be missing not because of the value itself or the value of any other variable but for no particular reason. People just forget to fill in some values. This is usually rarely the case and it's important to investigate why values are missing.

### Deletion   
Deletion is one way to handle missing values. It is the easiest but not the best way. One way to delete is *column* deletion: if a variable has too many missing values, just delete the variable. For example, in the above table more than 50% of the samples have Marital status missing and it might be tempting to remove that column all together. However, this may negatively impact the accuracy of the model because usually marital status is correlated to buying a house, e.g. married couples are more likely to purchase than single people.

Another way to delete missing values is *row* deletion: remove the sample that has a missing value altogether. This method can work if the missing values are MCAR and the number of samples with missing values is small, such as **less than 0.1%**. Row deletion is not effective if you have to throw out 10% of your data. However, if the missing data is not completely at random, removing those samples will affect the model performance. For instance removing the rows with missing annual income will be bad because having the income column missing likely correlates to buying a house which is what we want to predict. Also, if the missing values are at random, removing those samples can create biases in the model. For example, if we remove all the rows of gender A which have their age missing, our model will be biased towards gender B and not able to make good predictions for gender A correspondents.
### Imputation
Imputation means filling in the missing values with certain values. How do we figure out what these certain values are?   
1. Fill in with defaults, e.g. if the job is missing fill it with `""` 
1. Fill with the mean, median or mode. For example, if the temperature value is missing for a data sample with month value July, fill it with the median value of July
Be aware of filling in missing values with other possible values because it can create very hard to discover bugs. For instance do not fill in missing children with 0 which will make it hard to distinguish between people with no children and people who we do not have information of how many children they have. Another problematic example is filling in missing age with 0 and passing that to a model which has never seen 0 as an age value and made unreasonable predictions.
### **Scaling**
Scaling is one of the easiest ways to give your model a performance boost. To the model numbers are just numbers, it doesn't matter what feature it corresponds to, e.g. salary or age and the higher numbers in the salary may be interpreted as more important than lower numbers in a completely different feature, age for example. So it's important to scale your features before inputting them to models. Here are some common feature scaling methods:

1. Normalisation: Scales the features to be in the range of [0, 1]:

    $x' = \frac{x - min(x)}{max(x) - min(x)}$

    The author emprically has found that scaling to be in [-1, 1] range works better. Here is how you would scale the features to be within an arbitrary range of [a, b]:

    $x' = a + \frac{(x - min(x))(b - a)}{max(x) - min(x)}$

    Scaling to an arbitrary range works well when you don't want to make any assumptions about your variables5
1. Standardisation: If you think that your features might follow a normal distribution, it might be useful to scale them to have zero mean and unit variance:

    $x' = \frac{x - \bar{x}}{\sigma}$ 

1. Log transformation: In practice ML models struggle with skewed distribution of features. This method doesn't work in all cases and you should be cautious of analysis performed on log-transformed data instead of original data: $x' = log(x)$
### **Discretisation**
Based on the author's experience discretisation rarely helps. It is the process of converting continuous features into discrete features. It is also known as quantisation and binning. It is done by creating buckets of given values and grouping all features with values within a certain range to that bucket. Theoretically, the point is that for a feature annual income, we want the model to treat $5900.5 the same as $60000. But the model sees them as different numbers and treats them differently. Assume we make three buckets for this feature:   
* Lower income: less than $35,000/year
* Middle income: between $35,000 and $100,000/year
* Upper income: more than $100,000/year

We can do the same with discrete features, e.g. age groups.    
The downside of discretisation is that it introduces discontinuities at the category boundaries - $34,999 is now different from $35,000 which is treated the same as $100,000. For choosing the boundaries plotting histograms, basic quantiles, and some subject matter expertise can help.
### **Encoding Categorical Features**
Categorical features are not always static, e.g. brands in a recommendation system, accounts in spam detection. If we train on a fixed range of indexes corresponding to all the categories existing in the training set, your system will break when an unseen category shows up. For example, consider an e-commerce recommendation engine where the training data has 2,000,000 brands so you encode them from 0 to 1,999,999. Then you push to production and get complaints because a certain unseen brand does not get any of its products recommended by your model because it is encoding it as UNKNOWN. You can change the behaviour by encoding the top 99% most popular product and encoding the least popular products as UNKNOWN. This way the model at least knows how to deal with UNKNOWN brands.    
Now the model works but the CTR on recommendations plummet. Because over the last hour 20 new brands joined your site with a mix of luxury and knock-off brands but your model treats them all as unpopular brands.   
This is a very common problem in many fields and use cases of ML in production. But how do you go about putting new user accounts into different groups?   
One solution to this is the *hashing trick*, popularised by the package Vowpal Wabbit developed at Microsoft. In this approach, you use a hash function to generate a hashed value of each category. The hashed value will become the index of that category. Because you can specify the hash space, you can fix the number of encoded values for a feature in advance, without having to know how many categories there will be. For example, if you choose a hash space of 18 bits, which corresponds to $2 ^{18} = 262,144$ possible hashed values, all the categories, even the ones your model has never seen before, will be encoded by an index between 0 and 262,144. 
### **Feature Crossing**
Feature crossing is the process of combining two or more features to generate new features. It is useful to model the nonlinear relationships between features and is therefore essential for models that can't learn or are bad at learning non-linear relationships, e.g. linear regression, logistic regression, and tree-based models. In addition, it occasionally helps neural networks learn nonlinear relationships faster. DeepFM and xDeepFM are the family of models that have successfully leveraged explicit feature interactions for recommender systems and CTR prediction.    
Feature crossing has some downsides:   
1. Your feature spaces blows up, because now you have multiples of the original features which requires a lot more data to learn all the possible values
1. Without more data the increased number of features leads to overfitting
### **Discrete and Continuous Positional Embeddings**
Embedding the positions is needed for when you want to process the inputs in parallel for the model to have a sense of the order of inputs. They can be learned or fixed using a predefined function usually sine and cosine. Fixed positional embeddings is a special case of what is called Fourier features. 
## Data Leakage
Data leakage is when some form of label leaks into the feature set and is used for making predictions and this information is not available during inference. Data leakage is challenging because it is usually not obvious that it's happening especially because your evaluation can gain great results but in production the model might fail in unexpected ways. Here are two examples of data leakage where the model performs well during evaluation but failed in production:   
1. Assume the scenario where patient chest scans are used to predict whether they have COVID-19 or not. The data has a mix of scans where the patient is standing or lying down. Because most patients with severe conditions took their scan lying down, the model learns to predict serious COVID risk from the person's position
1. Consider the same scenario as above. The model might pick up on the text font that certain hospitals use, e.g. fonts from the hospital with more serious cases becomes a predictor
1. Consider a model that predicts cancer on CT scans. Your data comes from hospital A and the evalution on data from this hospital is good. But when you evalutate your model using data from hospital B, the results aren't good. Turns out that patients in hospital A whome their doctor suspects cancer for them get scanned with specific machine whereas patients in the other hospital are assigned a machine by random. The labels have leaked into the features in this case as the model learns to make predictions based on the information on the scan machine.  
### **Common Causes**
### Splitting time-correlated data randomly instead of by time   
Time-correlated data means that the time the data is generated affects its label distribution. It is very important to not randomly split your data into train, validation and test splits in this case. Doing so allows your models to cheat during evaluation by having information about the future. We say that information from the future has leaked into the training process. Here are two examples.
1. Consider predicting stock prices, generally similar stocks e.g. tech stocks move together. So if 90% of stocks go down today, it's likely that the other 10% go down too. If you have data from 7 days and randomly split your data, prices from day seven can be included in your train split and inform the model about the market condition of that day. In this case your model has information about the future and if the rest of the stock prices from the seventh day are in the test split, the model will perform well due to the leak. You need to split by time, e.g. train on the first 6 days and predict on the seventh day.
1. Here's a less obvious example; music recommendations. The recommended songs depend on many factors other than the user's taste. For example the general music trend of the day. If a singer passes away for instance, it is much more likely for their songs to be listened to. If you include samples from a certain day in the train split, information about the music trend that day will be passed to your model making it easier to make predictions on **other samples on that same day**.
### Scaling before splitting   
As mentioned earlier in the chapter, scaling features is important. To scale the features, you need global statistics, e.g. mean and variance. One source of information leakage is computing the statistics on the *entire* training data, i.e. before splitting. This leaks information about the test data's statistics into the training process and allows the model to **adjust its predictions for the test samples**. Because this information is not available during inference, the model's performance will likely degrage.    
To avoid this, first split your data then scale, i.e. use the statistics of the train split to scale **all the splits**. Some even suggest creating splits before EDA to not gain any information about the test split accidentally. 

### Filling in missing data with statistics from the test split   
Similar to scaling before splitting, if you use the statistics of the entire dataset to determine the values you want to fill the missing values with, information about the test split will leake into the train set.   
To mitigate this issue, split the data first, then use the statistics from the train split to determine the filling values *in all splits*.
### Poor handling of data duplication before splitting
Duplicate or near-duplicate data and failing to remove them before splitting can result in having the same data in train and validation/test splits. Duplication can be a result of data collection, merging different data sources or from data processing, e.g. oversampling.   
To avoid this always check for duplicates before splitting and also after splitting just to make sure. Also, if you are oversampling do it after splitting.
### Group leakage   
If a group of examples have strongly correlated labels but are divided into different splits, you have group leakage. For example, if a patient gets two CT scans 1 week apart. The labels are likely the same and if one is in the training and the other in test, your model cheats.   
It's hard avoiding this type of leakage without understanding how your test is split.
### Leakage from data generation process
Leakage can be a result of the data collection itself. The third example at the beginning of this section with the model relying on the type of scanning machines is an example. It's really hard to detect this type of leakage because you have to know how the data is collected and in this example that the procedure is different between hospitals A and B. It's a good idea to normalise all the images to have the same resolution to make it harder for the model to know which image is from which machine.   
Keeping track of the sources of your data and understanding how it is collected and processed helps with identifying these leakages. Include subject matter expertise who know about the data collection process.
### Detecting Data Leakage
Data leakage can happen in any step of the ML pipeline from collecting, sampling, splitting to feature engineering. It is therefore important to monitor for leakage during the entire ML lifecycle. Here are some tips for detecting data leakage:
1. Measure the correlation between each feature and the label. If a feature has unusually high correlation, investigate how this feature is generated and whether the correlation makes sense. Note that its possible for individual features to not contain leakage but a combination might contain leakage. For example, for a model predicting how long an employee will stay at a company the start date and end date separately may not be telling but both together can give us more information.
1. Do ablation studies to measure how important a feature or a set of feature is to your model. If removing a feature deteriorates the performance significantly investigate why that is. With thousands of features, doing this analysis on each possible combination is not feasible, but with subject matter expertise it will be useful to do it on a subset that you are most suspicious about. Also, these analysis can run offline on your own schedule, so you can leverage your machines during downtime for this.
## Engineering Good Features
Having more features is not necessarily a good thing for the reasons below:
1. Has the risk of overfitting
1. Increases memory requirements to serve the model, which in turn requires more expensive machine/instance to serve the model
1. More features means more opportunity for data leakage
1. Increased number of features can lead to increased inference latency when doing online prediction especially if the features need to be extracted from raw data for online predictions
1. Useless features become technical debts. Whenver the data pipeline changes, all the affected features need to be adjusted accordingly. For example, if you application decides to no longer take the user's age all the features (including the useless ones) need to be updated

In theory less important features should get 0 weight with regularisation methods, e.g. L1. However, in practice it makes models learn faster if features that are no longer helpful are removed. Below are two factors you might want to consider when evaluating whether a feature is good for a model. 
### **Feature Importance**
If you are using classical ML algorithms like boosted gradient treest, the easiest way is to used the built-in feature importance functions. For model-agnostic methods SHAP and the open-source library InterpreML are useful.
### **Feature Generalisation**   
The goal of training an ML model is to make correct predictions on unseen data. For this to be possible the features need to generalise to unseen data. For example in predicting whether a comment is spam or not using the post identifier is not generalisable and should not be used. However, the identifier of the user who posts comments such as usernam can be useful.   
To measure feature generalisation two aspects should be considered; feature coverage and distribution of feature values.   

### Coverage
A features coverage is the percentage of samples that have a value for that feature. What coverage percentage is good for a feature to be generalisable depends on the feature and why the values are missing. If you want to predict who will buy a home in the next 12 months and think that number of children is a good feature but only 1% of you data has a value for that feature, it might not be useful. However, if only 1% has values and 99% of examples with this feature have POSITIVE labels, this feature is useful. Also, if features are not missing at random, having a feature or not may be a strong indicator of its value and you should consider it.

Coverage can differ widely between slices of data and even in the same slice of data over time. If coverage of a feature differs a lot between the train and test split, e.g. 90% coverage in train and 20% in test, this can indicate that your splits come from different distributions. You should investigate whehter the way you split makes sense and whether this feature is a cause for data leakage.

### Distribution
For features that have values, it's useful to look at there distribution. If the set of feature values that appears in the seen data has no overlap with the values of test, this feature might be hurt performance.   
For example, for a model predicting ETA of rides, consider a feature DAY_OF_THE_WEEK. This feature can be useful as there is more traffic on weekdays. Assume that this feature has 100% coverage but the train split includes days Monday through Saturday and the test split includes Sunday as the feature value. If you don't have a good scheme to encode the days, the feature won't generalise to the test split and it may harm your model's performance.   
When considering the feature's generalisation, there's a trade-off between generalisation and specificity. You might realise that traffic during an hour only changes depending on whether that hour is the rush hour. So you generate a new boolean feature, IS_RUSH_HOUR which is more generalisable but less specific than a feature that indicates the hour itself. Using only IS_RUSH_HOUR though might cause the model to lose important information about the hour.

