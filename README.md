# Recommender-System-Codenation-Project  
### See the whole process and notes in the [Main Notebook](https://github.com/Key0412/Recommender-System-Codenation-Project/blob/master/main.ipynb)!

### <a name="objective">1 Objective</a>

This project's objective is to provide an automated service that recommends new business leads to a user given his current clients portfolio. It was created as the final challenge in the context of the acceleration in Data Science by [Codenation](https://www.codenation.dev/) (now [Trybe](https://www.betrybe.com/)).  
For this project, a ***Content Based Filtering Recommender System* based in *Logistic Regression predicted probabilities*** is going to be used. 

### <a name="dataset">2 The Dataset</a>

* `estaticos_market.csv`: .csv file, it's compressed file (.zip format) is available at the project's github repo. contains IDs for 462298 companies. Contains 181 features, including id. Eventualy refered as complete/whole dataset or market database.
* `estaticos_portfolio1.csv`: .csv file, contains clients' ids for the owner of portfolio 1. Contains IDs for 555 client companies. It also has 181 features, including ids, which are the same as in the complete dataset.
* `estaticos_portfolio2.csv`: .csv file, contains clients' ids for the owner of portfolio 2. Contains IDs for 566 client companies.  
* `estaticos_portfolio3.csv`: .csv file, contains clients' ids for the owner of portfolio 3. Contains IDs for 265 client companies.

### <a name="approach">3 Selected Approach and Steps</a>

After experimentation, research and input from felow students and the community, **For this project, a Content Based Filtering Recommender System based in *Logistic Regression* is going to be used**. I call it recommender system in a generalist manner, you'll see it's not quite like the examples of recommender systems around (and I don't mean more complex or smart or even good), but it does recommend leads! It's not quite a recommender system per se (see the references in the jupyter notebook), e.g. it does not uses technologies as TF-IDF, Matrix Factorization, similarity comparison through euclidean/cosine distances, but it does recommend leads!  
The steps taken, overall, are:  
* The companies that are already clients can be used as targets provided they're encoded as 0s and 1s, or False and True.  
* The processed database can be used as predictors.  
* We aim not to obtain the predictions per se, but the logistic regression predicted probability that the company is indeed a client. With this, we can sort the companies recommended based on the predicted probability that they're clients.  
* Since there's almost 470E3 companies, we'll use KMeans clustering to group the companies and train logistic regressions for each group.  
* The data is very imbalanced - each portfolio have around 500 companies, and we've just cited the size of the dataframe with all companies. We'll use SMOTE oversampling along the training sets to address this issue.  
* Metrics will be shown for the trained logistic regression on portfolio 2 - then, recommendations will be made for all portfolios.
* The recommendations made will be evaluated through the MAP@k metric.  

### <a name="results">4 Results</a>

**The average precision is highest at top 3 and goes down as the threshold increases - we're indeed recommeding better Leads at the beggining of the list, mainly for the top 3, 5 and 10 recommendations!**  

`MAP@3: 1.0, MAP@5: 0.8, MAP@10: 0.569048, MAP@25: 0.40315, MAP@50: 0.348891, MAP@100: 0.333344 MAP@500: 0.250661, MAP@1000: 0.39807`  
  
___
#### [Youtube video with high level explanation - PTBR](https://www.youtube.com/watch?v=mPy3HNEKsns&feature=youtu.be)
___