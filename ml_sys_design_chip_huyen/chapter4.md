# Training Data
## Sampling
Sampling is useful when you don't have access to all the data. It is also useful for cases where you don't have enough time to analyse the entire dataset and want to run a quick experiment on a subset of data.   
Sampling happens in many steps of the machine learning lifecycle. For example, for creating the training data, data points from all real-world data are sampled. These data points are sampled to make training, validation and test splits. After deploying, data from the set of events are sampled for monitoring. It's important to know different methods and how they work to avoid sampling biases. There are two sampling categories, nonprobability sampling and random sampling.
### Nonprobability Sampling
When the selection of data is based on any probability criteria, for example:   

**Convenience Sampling**   
Data points are selected based on their availability.   

**Snowball Sampling**   
Existing samples determine future samples. For example, to scrape legitimate Twitter accounts without having access to Twitter's database, you start with a number of accounts and continue to scrape all accounts that they follow and so on.   

**Judgement Sampling**   
Domain experts decide what samples to include.   

**Quota Sampling**   
Samples are selected without randomisation from slices of the data based on quotas. For example, for a survey you want 100 responses from people under 30, between 30 and 60 and above 60 regardless of the distribution of each age group.   

Nonprobability sampling is not a good choice for collecting training data as it has selection bias. However, mostly convenience sampling is used for training machine learning models, e.g. for training sentiment analysis, IMDB and Amazon reviews are used which has bias towards those who are willing to leave reviews, have internet access, etc. and therefore is not representative of the population. Especially when these trained models are used for other sentiment analysis tasks.   
This method, is a quick and easy way to get started on your project, but probability-based sampling is more reliable.

### Simple Random Sampling
This is the simplest form of random sampling where each data point has an equal probability of being selected. The advantage of this method is that it is easy to implement; however, in the case of minority data groups the chances of being selected is very low. Models trained on this data may think the the minority group does not exist. 
### Stratified Sampling
To avoid the issue of simple random sampling. In this sampling method you first split your data into groups (each group is called a stratum) and then you sample from each stratum. This way the rare classes will be included in the samples. For example, if you want to sample 1% of the data with two groups A, B. You can sample 1% of stratum A and 1% of stratum B.   
An issue of this method is that it is not always possible split samples into groups, especially in multilabel tasks where one sample can belong to more than one group.
### Weighted Sampling
In this method, each sample is given a weight which determines the probability of it being selected. This is useful as it allows you to use your domain knowledge about the real world data. For example, if you know that more recent data is more important you can give it a higher weight and therefore a higher chance of getting sampled relative to older data.   
Weighted sampling is closely related to sample weights. In weighted sampling samples are weighted to be selected for creating training data, whereas in sample weights, samples in the training data are weighted to signify their importance. For example, you may weight the samples from a rare class higher to affect the loss function more and therefore change the decision boundary in such a way that the rare class is classified more accurately.
### Reservoir Sampling
This method is useful for sampling from streaming data. Assume that you have an incoming stream of tweets and you want to sample k tweets from it. You want each tweet to have equal probability of being selected but you don't know in advance how many tweets there will be in total to know the probability of each selection. Also, if you stop the sampling at any time you want all the tweets to have been selected with the correct probability. Reservoir sampling is the method that satisfies these criteria. It involves a reservoir which can be an array and consists of three steps:   
1. Put the first k elements into the reservoir.
1. For each incoming nth element, generate a random number i such that 1 $\leq$ i $\leq$ n.
1. If 1 $\leq$ i $\leq$ k: replace the ith element in the reservoir with the nth element. Else, do nothing.

This means that each incoming nth element has $k/n$ probability of being selected to be in the reservoir. You can also prove that each element in the reservoir has $k/n$ probability of being there. This means that all samples have equal probability of being selected and if we stop the selection at any time, all samples in the reservoir have been selected with the correct probability. The figure below illustrates how this sampling method works.
<center>
<img src="images/reservoir.jpg" width="60%" alt="reservoir" title="reservoir">
</center>

### Importance Sampling
This method is very useful as it allows us to sample from a distribution when we don't have access to it but only have access to another distribution. Suppose you want to sample $x$ from a distribution $P(x)$ which is infeasible to sample from. There is another distribution $Q(x)$ which is much easier to sample from. You can sample $x$ from $Q(x)$ and weigh it by $P(x)/Q(x)$. $Q(x)$ is called the *proposal distribution* or the *importance distribution* and can be any distribution as long as it is positive wherever $P(x)$ is non-zero. The following equation shows that in expectation, $x$ sampled from $P(x)$ is equal to $x$ sampled from $Q(x)$ weighted by $P(x)/Q(x)$:   
$E_P[x]=\sum_{x}P(x)x = \sum_{x} Q(x)x\frac{P(x)}{Q(x)} = E_Q[x\frac{P(x)}{Q(x)}]$

One use case of importance sampling is policy-based reinforcement learning. When updating the policy, the value functions of the new policy have to be estimated, but calculating the total rewards of taking an action can be costly because it requires considering all possible outcomes until the end of the time horizon after that action. However, if the new policy is relatively close to the old policy, you can calculate the total rewards based on the old policy and weight them according to the new policy. The rewards from the old policy make up the proposal distribution.
## Labeling
### Hand Labels
Manually labeling data points for creating training data can be problematic for a number of reasons:   
1. It's expensive. It can be very costly to have multiple annotators label your data especially if subject matter expertise is needed, e.g. board-certified radiologist.
1. Data privacy can be an issue. If you have sensitive data that needs labeling, it may not be possible to send data to annotators.
1. Data labeling is slow. It can take months for sufficient amount of data to be labeled which restricts you ability to adapt your model to changes in the real world. For example, consider a sentiment analysis model on a e-commerce platform that has been trained on annotated comments with POSITIVE or NEGATIVE labels. Assume that after launching, the PR team decides that they should care more about the angry comments and want a third label predicted, ANGRY. In this case, you would need to relabel your data which is not an easy process.
1. **Label multiplicity** : Another problem with hand labels is that different annotators might label the same data point differently. This is called data multiplicity. One way to mitigate it is to clearly define the labeling task, think of edge cases and communicate this with the annotators before they start labeling data.
1. **Data lineage**: Indiscriminately using data from multiple sources, generated from different annotators, without examining their quality can cause your model to fail mysteriously. It's good practice to keep track of the origin of each of your data samples as well as its labels, a technique called data lineage. It helps identify potential biases in the data and debug the model.
### Natural Labels
In some machine learning systems the model's predictions can automatically be evaluated (could be partial evaluation) by the system. An example is a model that estimates the travel time on Google Maps. If the user takes the suggested route, by the end of the trip, Google Map knows how long the trip actually took and can use this natural ground truth label to improve its model.    
Even if the task doesn't have inherently natural labels like the example above does, it can be be represented to the user in such a way to allow for feedback. For example, in a news feed app, allow for likes. 

**Feedback loop length**   
The amount of time it takes between the model prediction to receiving feedback is the feedback loop length, some products such as recommender systems in e-commerce websites have a short feedback loop, e.g. minutes. Others like Stitch Fix that recommends clothes and then sends them to the users have weeks long loops.   
There are also different types of user feedback that can be collected depending on the task and the objective. In an e-commerce website for example, clicking on the recommendation is one type of feedback, adding it to the cart is another and checking it out is an even stronger feedback. The clicks have a shorter feedback loop length and are higher in volume than checkout. Optimising for clicks therefore provides more retraining data, but you can argue that optimising for checkouts is more in line with the business objective of increasing revenue. There is no right or wrong answer here.   
Another aspect of the feedback loop is its length. How long should we wait to see if the recommendations were clicked on? If it's too short we might miss the clicks that happened after the feedback loop window ended and get false negative labels. But it allows you to capture labels faster and address issues with your model quicker. No matter how long your window length, there might be some premature negative labels.
### Handling Lack of Labels
**Weak Supervision**   
**Semi-supervision**   
**Transfer learning**   
**Active learning**
## Class Imbalance

**Challenges**   
**How to handle it?**
## Data Augmentation
### Simple Label-Preserving Transforms
### Perturbation
### Data Synthesis