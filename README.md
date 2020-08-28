# Recommender-System-Codenation-Project  
### See the whole process, analysis and notes on the [Projects' GitHub Page](https://key0412.github.io/Recommender-System-Codenation-Project/)!

### <a name="objective">1 Objective</a>

This project's objective is to provide an automated service that recommends new business leads to a user given his current clients portfolio. It was created as the final challenge in the context of the acceleration in Data Science by [Codenation](https://www.codenation.dev/) (now [Trybe](https://www.betrybe.com/)).  
For this project, a ***Content Based Filtering Recommender System* based in *Logistic Regression predicted probabilities*** is going to be used. 

### <a name="dataset">2 The Dataset</a>

* `estaticos_market.csv`: .csv file, it's compressed file (.zip format) is available at the project's github repo. contains IDs for 462298 companies. Contains 181 features, including id. Eventualy refered as complete/whole dataset or market database.
* `estaticos_portfolio1.csv`: .csv file, contains clients' ids for the owner of portfolio 1. Contains IDs for 555 client companies. It also has 181 features, including ids, which are the same as in the complete dataset.
* `estaticos_portfolio2.csv`: .csv file, contains clients' ids for the owner of portfolio 2. Contains IDs for 566 client companies.  
* `estaticos_portfolio3.csv`: .csv file, contains clients' ids for the owner of portfolio 3. Contains IDs for 265 client companies.

### <a name="approach">3 Selected Approach and Steps</a>

After experimentation, research and input from felow students and the community, **For this project, a Content Based Filtering Recommender System based in *Logistic Regression* is going to be used**. It's not quite a recommender system per se ([at least not like the ones I found](https://key0412.github.io/Recommender-System-Codenation-Project/#refs)), e.g. it does not uses technologies as TF-IDF, Matrix Factorization, similarity comparison through euclidean/cosine distances, but it does recommend leads!  
The steps taken, overall, are:  
* The companies that are already clients can be used as targets provided they're encoded as 0s and 1s, or False and True.  
* The processed database can be used as predictors.  
* We aim not to obtain the classifications per se, but the logistic regression predicted probability that the company is indeed a client. With this, we can sort the recommended companies based on the predicted probability that they're clients.  
* Since there's almost 470E3 companies, we'll use KMeans clustering to group the companies and train logistic regressions for each group.  
* The data is very imbalanced - each portfolio has around 500 companies, and we've just cited the size of the dataframe with all companies. We'll use SMOTE oversampling along the training sets to address this issue.  
* Metrics will be calculated for the trained logistic regression and recommendations made using portfolio 2.
* The recommendations made will be evaluated through the MAP@k metric.  

### <a name="results">4 Results</a>

MAP@K metrics for k=5, 10, 25 and 50:  
`MAP@5: 0.543333, MAP@10: 0.707103, MAP@25: 0.596837, MAP@50: 0.5175`

**Notes on Results for Portfolio 2**
* The MAP@5 metric means we're getting the first value wrong, while the remaining 4 values were indeed already clients.
* The MAP@10 metric means we're getting way more recommendations that were already of interest in this interval, in fact, we're missing out only the first item. See the list comparation below, it confirms the last sentence.  
`For the ten first recommendations, are they of interest (e.g. already clients)? 
Ordered recommendations : [False, True, True, True, True, True, True, True, True, True]`  
* The MAP@25 and MAP@50 metrics seem consistent, and even with 50 recommendations, we can present good recommendations. From the first 25 and 50 recommendations, 19 and 34 were already clients, respectivelly.

**With these results, it's possible to say we're recommeding good Leads at the beggining of the list, mainly for the top 5, 10 and 25 recommendations, while there were a significant number of interesting recommendations among the first 50 as well!**

___  

#### [Youtube video with high level explanation - PTBR](https://www.youtube.com/watch?v=mPy3HNEKsns&feature=youtu.be)
[![Recommender System to Generate Leads based on Clients' Portfolio video, Miniature Photo by Jamie Street on Unsplash](docs/video_thumbnail.png)](https://www.youtube.com/watch?v=mPy3HNEKsns&feature=youtu.be "Recommender System to Generate Leads based on Clients' Portfolio video, Miniature Photo by Jamie Street on Unsplash")  

___  