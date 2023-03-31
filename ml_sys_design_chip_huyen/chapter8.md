# Data Distribution Shifts and Monitoring
## Causes of ML System Failures
### *Software System Failures*
Similar to traditional software, in ML systems we care about operations metrics. Here are some examples:
1. Dependency Failure: This occurs when a software package or codebase that your system depends on breaks, in no longer maintained, etc.
1. Deployment Failure: Occurs due to errors during deployment. For example, you deploy the wrong model or the system does not have the correct read/write permissions with certain files.
1. Hardware Failure: The hardware you deploy your model, e.g. CPU and GPU breaks.
1. Downtime or crashing: If a component of your system runs from a server somewhere, such as AWS or a hosted service, and that server is down, your system will also be down.
### *ML-Specific Failures*
Here are some examples specific to ML systems that can cause failures in production:    
### Production data differing from training data
The assumption that the unseen data will be from the same distribution as the training data is practically incorrect, for two reasons:    
1. The real-world (unseen) data is infinite while the training data is finite and constrained by time, compute and human resources available during dataset creation and processing. There are many selection and sampling biases that make training data diverge from real-world data. When the model performs well in development but poorly when deployed, it is known as **train-serving skew**. 
1. Real world is not stationary. Preferences, social norms, trends change. These changes can be sudden, e.g. a celebrity endorsing your product which results in a sudden increase of new users, or it can be gradual or even seasonal. For example, in winter people are more likely to request ride shares than in spring.    


However, due to poor practices in deploying ML systems, a large percentage of what might look like data shifts on monitoring dashboards are caused by internal errors, such as bugs in the data pipeline, training feature extraction not being correctly replicated during inference, missing values incorrectly inputted, features standardised using statistics from the wrong subset of data, wrong model version or bugs in the app interface that force users to change their behaviors. 
### Edge cases
Edge cases are the data samples so extreme that they cause the model to make catastrophic mistakes. Edge cases generally refer to data samples drawn from the same distribution. However, if there is a sudden increase in the number of samples that your model doesn't perform well, it could be an indication that the underlying data distribution has shifted. 

*Outliers versus edge cases*
In this book, outliers are refer to the data, an example that differs significantly from other data samples. Edge cases are related to performance, i.e. an example where the model performs poorly on. Outliers can lead to edge cases but not necessarily. For example, a person jaywalking is an outlier but if your self-driving car correctly identifies them as a pedestrian and takes appropriate action, it is not an edge case.

### Degenerate feedback loops
In the case of natural labels, the ML suggestions can influence the feedback which in turn influences the next batch of training data fed into the model. This in turn influences the system's future outputs. Here are two examples:    
1. A model recommends songs that the users are most likely to click on. The songs are ordered by the model's ranking and so the most confident predictions are placed higher. This position makes the the higher placed song get more clicks which can incorrectly be interpreted as good recommendations.    
Degenerate feedback loops are one reason why popular books, movies or song keep getting more popular which makes it hard for new items to break into popular lists. This is known as exposure bias, popularity bias, filter bubbles, and echo chambers.
1. A model that grade resumes finds a certain feature X to be predictive of the candidate's competence. The recruiter will keep interviewing people with that feature that the model recommends which gets fed back to the model as confirmation that feature X is predictive and gives it higher weight. Visibility into how the model makes predictions - such as measuring the importance of each feature for the model - can help detect bias toward feature X in this case.

**Detecting degenerate feedback loops**
1. Measuring the popularity diversity of a system's outputs even when the system is offline. An item's popularity can be measured by how many time it has been interacted with, e.g. liked, seen, bought in the past. The popularity of all items will likely follow a long-tail distribution as only a small number of items are interacted with a lot. Some metrics such as *aggregate diversity* and *average coverage of long-tail* items can be used to measure the diversity. Low scores means the model outputs are homogeneous which might be caused by popularity bias.
1. Measure the hit rate against popularity: Bucket the items based on level of interaction, e.g items which are interacted with over 1000 times goes in one bucket. Then the prediction accuracy of a recommender system for each bucket is measured. If the accuracy is much better for the popular bucket, it likely suffers from popularity bias.

**Correcting degenerate feedback loops**
1. Introducing randomisation: This is how TikTok evaluates videos, by introducing them to a small pool of traffic. If the engagement is good, they introduce it to a bigger pool. However, this has the risk of poor user experience as the users may not like totally random suggestions.
1. An intelligent exploration strategy, e.g. contextual bandits as an exploration strategy, can help increase the diversity with acceptable prediction accuracy loss. 
1. Positional features: In the case of the song recommender mentioned earlier, if the users click on the top song more it is difficult to know whether the top song is actually the best prediction or it's because of its position and any song in that position will get clicked out. In this case a binary feature of whether the song is in the 1st position or not will be useful during training. For inference, you want to predict whether a user will click on a song regardless of its position so you can set the positional feature to False and rank the results. A more sophisticated approach would be to use two models. The first predicts the probability that the user will see and consider a recommendation taking into account the position at which that recommendation will be shown. The second model then predicts the probability that the user will click on the item given that they saw and considered it and does not consider the position at all.
## Data Distribution Shifts
### *Types of Data Distribution Shifts*
### *General Data Distribution Shifts*
### *Detecting Data Distribution Shifts*
### *Addressing Data Distribution Shifts*

## Monitoring and Observability
### *ML Specific Metrics*
### *Monitoring Toolbox*
### *Observability*