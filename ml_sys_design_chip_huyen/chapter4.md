# Training Data
## Sampling
Sampling is useful when you don't have access to all the data. It is also useful for cases where you don't have enough time to analyse the entire dataset and want to run a quick experiment on a subset of data.   
Sampling happens in many steps of the machine learning lifecycle. For example, for creating the training data, data points from all real-world data are sampled. These data points are sampled to make training, validation and test splits. After deploying, data from the set of events are sampled for monitoring. It's important to know different methods and how they work to avoid sampling biases. There are two sampling categories, nonprobability sampling and random sampling.
### Nonprobability Sampling
When the selection of data is based on any probability criteria, for example:   

**Convenience Sampling**   
Data points are selected based on their availabiltiy.   

**Snowball Sampling**   
Existing samples determine future samples. For example, to scrape legitimate Twitter accounts without having access to Twitter's database, you start with a number of accounts and continue to scrape all accounts that they follow and so on.   

**Judgement Sampling**   
Domain experts decide what samples to include.   

**Quota Sampling**   
Samples are selected without randomisation from slices of the data based on quotas. For example, for a survey you want 100 responses from people under 30, between 30 and 60 and above 60 regardless of the distribution of each age group.   

Nonprobability sampling is not a good choice for collecting training data as it has selection bias. However, mostly convenience sampling is used for training machine learning models, e.g. for training sentiment analysis, IMDB and Amazon reviews are used which has bias towards those who are willing to leave reviews, have internet access, etc. and therefore is not representative of the population. Especially when these trained models are used for other sentiment analysis tasks.   
This method, is a quick and easy way to get started on your project, but probability-based sampling is more reliable.

### Simple Random Sampling
This is the simplest form of random sampling where each data point has an equal probability of being selected. The advantage of this method is that it is easy to implement; however, in the case of minority data groups the chances of being selected is very low. Models trained on this data may think tha the minority group does not exist. 
### Stratified Sampling
To avoid the issue of simple random sampling. In this sampling method you first split your data into groups (each group is called a stratum) and then you sample from each stratum. This way the rare classes will be included in the samples. For example, if you want to sample 1% of the data with two groups A, B. You can sample 1% of stratum A and 1% of stratum B.   
An issue of this method is that it is not always possible split samples into groups, especially in multilabel tasks where one sample can belong to more than one group.
### Weighted Sampling
In this method, each sample is given a weight which determines the probability of it being selected. This is useful as it allows you to use your domain knowledge about the real world data. For example, if you know that more recent data is more important you can give it a higher weight and therefore a higher chance of getting sampled relative to older data.   
Weighted sampling is closely related to sample weights. In weighted sampling samples are weighted to be selected for creating training data, whereas in sample weights, samples in the training data are weighted to signify their importance. For example, you may weight the samples from a rare class higher to affect the loss function more and therefore change the decision boundry in such a way that the rare class is classified more accurately.
### Reservoir Sampling

### Importance Sampling
## Labeling
### Hand Labels
**Label multiplicity**   
**Data lineage**
### Natural Labels
**Feedback loop length**
### Handeling Lack of Labels
**Weak Supervision**   
**Semi-supervision**   
**Transfer learning**   
**Active learning**
## Class Imbalance

**Challenges**   
**How to handle it?**
## Data Augmentation
### Simple Label-Preserving Transforms
### Pertubation
### Data Synthesis