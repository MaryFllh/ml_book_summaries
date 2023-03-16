# Model Development and Offline Evaluation
## Model Development and Training
Deep learning is finding more use cases in production, but that doesn't mean the classical ML algorithms are going away. Many recommendation systems still rely on matrix factorisation and collaborative filtering. Tree-based algorithms, e.g. gradient-boosted trees powers a lot of classification tasks with strict latency requirements.   
It is also common for deep learning models to be used in tandem with calssical models. For example, a k-means clustering model might be used to extract features to input into a neural network. 
### **Evaluating ML Models**
When selecting a model, it's important to select a set of candidate models that fit the problem. For instance, if you want a text classifier, you can look into Naïve Bayes, logistic regression, RNNs, and transformer-based models. For fraud detection, you want to look at KNNs, isolation forest, clustering and neural networks.   
In order to know what models to consider based on your problem, you need to have knowledge about common ML tasks and typical approaches.
**Tip**: To keep yourself up to date iwth new ML techniques and models, monitor trends at major ML conferences, e.g. NeurIPS, ICLR, ICML as well as following researchers whose work has a high SNR on Twitter.
### 1. Avoid the state-of-the-art trap
The goal is to find a solution that works. If that solution is a cheaper and simpler approach than the fancy state-of-the-art, so be it.
### 2. Start with the simplest models
Simpler is better than complex. Simplity serves three purposes:   
1. Simpler models are easier to deploy and deploying early allows you to verify that your predicition pipeline is consistent with your training pipeline quicker.
1. Starting simple and adding complexity step by step makes it easier to understand the model and debug issues.
1. The simplest models can be used as a baseline for your future complex models.
### 3. Avoid human biases in selecting models
The performance of a model architecture heavily depends on the context, data, hyperparameters, etc. it is really important to not be biased towards a certain architecure and spend more time running experiments with it while running fewer tests on other architectures. For example, if you want to compare a tree-based model with a pretrained BERT model, run 100 experiments on both and not 100 on BERT and a few on the tree-based because you really want the more sophistaced model to be the better performing one.
### 4. Evaluate good performance now versus good performance later
Models degrade and its important to keep in mind that the best-performing model architecture you go live with may not be the best 2 months after. For example, you might start with a tree-based model at first because you have little data, but after going to production and collecting data you are able to train a neural network that outperforms the tree-based model.  
Another example is recommendation systems, a collaborative filtering method vs a neural network. At first the collaborative filtering might outperform the NN. However, the NN is much more efficient at online training and can update itself with each incoming example. Whereas the collaborative filtering would need to look at all the data to update its underlying matrix. One solution to this is to go live with both, have the collaborativ filtering make recommendations to users, while the NN uses the data to improve its predictions. Once you have enough data passed through the NN, and it outperforms the collaborative filtering, you can replace it with the NN.
### 5. Evaluate trade-offs
There are many trade-offs to be considered when developing a machine learning model. The classic example is the false postive and false negative trade-off. For a finger authentication app you would want to have lower false positives, whereas in a COVID-19 screening you would want to have fewer false negatives.  
There is also accuracy and compute or accuracy and interpretability trade-offs you would need to consider.
### 6. Understand your model's assumptions
It's important to know what assumptions your model makes about the data and whether your data satisfies those assumptions. Here is a list of common assumptions:
* IID: Neural networks assume that the samples are independent and identically distributed, which means that all examples are independently drawn from the same joint distribution.
* Smoothness: Every supervised method assumes that there's a set of functions that can map the input to the output such that similar inputs are transformed into similar outputs. 
* Tractability: Let X be the input and Z the latent representation of X. Every generative model makes the assumption that it's tractable to compute the probability P(Z|X).
* Boundries: Linear classifiers assume that the decision boundries are linear
* Conditional independence: A naïve bayes assumes that the attribute values are independent of each other given the class.
* Normally distributed: many statistical methods assume that the data is normally distributed.  
### **Ensembles**
Ensembles have consistently proven to provide performance boosts. However, they are less favoured in production because it is harder to deploy and maintain.   
If each base learner is uncorrelated with the others, using an ensemble will give you a performance boost. For example assume you have three spam detector base models each have a 70% chance of of being correct for each sample. The overall accuracy of an ensemble is as the table below shows, 0343 + 0.441 = 0.784 or 78.4%.
<center>
<img src="images/ensemble.jpg" width="50%" alt="ensemble" title="ensemble">
</center>
The less correlation there is between base learners, the better the ensemble will be. Therefore, it's common to choose very different types of models for an ensemble, e.g. transformer, RNN and a gradient-boosted tree.        

There are there three ways to create an ensemble:    

1. Bagging: Bootstrap aggregating where different datasets are created with random selection with replacement (with replacement so that the bootstraps are created independently from each other). This method reduces variance and helps to avoid overfitting.  
Bagging generally helps with improving unstable methods such as neural networks, classification and regression trees, and subset collection in linear regression. However, it can mildly degrade the performance of stable methods like KNN. 
1. Boosting: Boosting is a family of iterative ensemble algorithms that convert weak learners to strong ones. Each learner is trained on the same dataset but the samples have different weights in each iteration. The final strong classifier is a weighted combination of the existing classifiers-classifiers with smaller training errors have higher weights.
1. Stacking: In stacking you train base learners from the training data then create a meta learner that combine the outputs of the base learners either by majority vote or training another model that takes the base learners' output as its input.
### **Experiment Tracking**
Here's a list of useful things to track during training experiments:   
1. The *loss curve* corresponding to the train split and each of the eval splits.
1. The *model performance metrics* that you care about on all the nontest splits.
1. The log of *corresponding sample, prediction, and ground truth label*. This is useful for ad hoc analytics and sanity check.
1. The *speed* of your model evaluated by the number of steps per second or, if your data is text, the number of tokens processed per second.
1. *System performance* metrics such as memory usage and CPU/GPU utilisation. They're important in identifying bottlenecks and avoid wasting system resources.
1. The values over time of any *parameter* and *hyperparameter* whose changes can affect your model's performance, such as learning rate if a learning rate schedule is used; gradient norms (both globally and per layer), especially if you're clipping your gradient norms; and weight norm, especially if you're doing weight decay.   
### **Experiment Versioning** 
Data versioning is like flossing, everyone agrees it's a good thing to do, but few do it. 
### **Debugging ML Models**
Here are some common failure points of ML models:
1. Theoretical constraints: Each model has some assumptions about the data and features it uses. It can be that the inputs do not actually satisfy these assumptions, e.g the decision boundry is not linear but a linear model is used.
1. Poor implementation of the model: The model is a good fit but ther are bugs in the implemenation. For example, in PyTorch you might have forgotten to stop the gradient updates during evaluation.
1. Poor choice of hyperparameters
1. Data problems: Data samples and labels might be incorrectly paired, noisy labels, features are normalised using outdated statistics, etc.
1. Poor choice of features: Too many features can cause overfitting and can also cause data leakage. Too few features might lack predictive power for your models.

Here are a few tips for approaching debugging deep learning models.
1. Start simple and gradually add more components: Start with the simplest model and slowly add more components to see if it helps or not. For example, when building an RNN start with just one layer before adding multiple and using regularisation. If you want to use a BERT-like model which uses both a masked language model and next sentence prediction loss, you might want to use only the MLM loss before adding the NSP loss.
1. Overfit a single batch: After you have a simple implementation of the model, try to overfit a small amount of training data and run evaluation on the same data to make sure it gets to the smallest possible loss. For example, in an image recognition task, overfit on 10 images and see if you can get the accuracy to be 100%. Or in a MT taks overfit on 100 sentence pairs and see if you can get the BLEU score of near 100. If you can't overfit on small amount of data, there might be something wrong with the implementation.
1. Set a random seed: There are so many sources of randomisation in your model, e.g. weight initialisation, dropout, data shuffling, etc. which makes it hard to compare results across different experiments-you don't know if the change in performance is due to a change in the model or a different random seed. Set a random seet to ensure consistency between different runs. It also allows you to reproduce errors and other people to reproduce your results.

### **Distributed Training**
### **Auto ML**
## Model Offline Evaluation
### **Baselines**
### **Evaluation Methods**