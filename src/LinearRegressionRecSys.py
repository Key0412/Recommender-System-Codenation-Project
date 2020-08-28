# Related third party imports
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


class LinearRegressionRecSys():
    """
    Class used to train logistic regression model in each defined cluster and to provide recommendations.
    """
    
    def __init__(self, portfolio, database, cluster_labels, random_state=None):
        """
        Set class variables and check if the database contains the portfolios' IDs.
        The database must contain IDs as index and the portfolio must contain `id` as a feature.
        :param portfolio: Pandas DataFrame, contains only the portfolio clients' IDs as feature `id`.
        :param database: Pandas DataFrame, contains all the companies' IDs as index.
        :param cluster_labels: Pandas DataFrame, contains numbers from 0 to max number of clusters. Maps each company to a cluster.
        :param random_state: integer, default=None, set random state of internal processess.
        """        
        # Set internal variables: portfolio, database, cluster_labels
        self.portfolio = portfolio
        self.database = database
        self.cluster_labels = cluster_labels
        self.random_state = random_state
        
        # Test - check if database contains portfolios' IDs
        print(f"\nTesting Portfolio . . .\n")
        print(f"Database size: {self.database.shape[0]}")
        print(f"Portfolio size: {self.portfolio.shape[0]}")
        assert np.all(self.portfolio["id"].isin(self.database.index)), "Not all the portfolios' ids are in the database"
        print("Portfolios' ids are in the database\n")
        # Set internal variable: rating_df Pandas DataFrame, with features:
        #   - id: the id of the company
        #   - client: if the company is a client (present) on the porfolio
        #   - cluster: to which cluster the company belongs
        rating_df = self.database.reset_index()["id"] # get all IDs
        portfolio_flag = rating_df.isin(self.portfolio["id"]) # True means it is a client
        portfolio_flag.name = "client"
        rating_df = pd.concat([rating_df, portfolio_flag, self.cluster_labels], axis=1) # concatenate IDs, client flag and cluster labels
        self.rating_df = rating_df        
        
    def _get_cluster_target_df(self, rating_df, cluster):
        """
        Returns a Pandas DataFrame with all companies present in the cluster and a pandas series that represents if the company is a client.
        :param cluster: integer, cluster from which predictors dataframe will be constructed.
        :create self._cluster_df: Pandas DataFrame with features and IDs of all companies present in the cluster. 
        :create self._target: Pandas Series that represents if the company is a client.
        """
        condition = rating_df["cluster"] == cluster # means that we're accessing the right cluster        
        cluster_ids = rating_df[(condition)]["id"] # gets ids from all companies in the cluster
        cluster_df = self.database.loc[cluster_ids, :] # get features from all companies in the cluster
        target = rating_df.loc[condition, "client"] # get target for cluster - True means it is a client
        self._cluster_df = cluster_df
        self._target = target
        
    def train_classifiers(self):
        """
        Train logistic regression classifier for each cluster present in the companies dataframe. \
Predictor is a dataframe with all companies features' for each cluster, target is Pandas Series with boolean values indicating if company is client.
        Does train test split, SMOTE oversampling, logistic regression training for each cluster.
        :create self.train_output: dictionary, contains keys:
            -"client_flag": 1 if cluster has no clients, 0 if has. 
            The following keys are present in the second case:
                -"classifier": trained logistic regression object.
                -"metrics": dictionary, contains keys:
                    -"accuracy": accuracy score
                    -"precision": precision score
                    -"recall": recall score
                    -"f1_score": f1 score
                    -"roc_auc": area under the curve
        """                
        n_clusters = self.cluster_labels.nunique()[0]

        train_output = {}

        for cluster in range(n_clusters):

            print(f"- Veryfing Cluster {cluster} -\n")
            self._get_cluster_target_df(self.rating_df, cluster)

            print(f"Cluster size: {self._cluster_df.shape[0]}")
            print(f"Clients in cluster: {self._target.sum()}")
            print(f"Clients per cluster ratio: {round(100*(self._target.sum()/self._cluster_df.shape[0]), 3)} % \n")

            print("Processing:\n")

            if self._target.sum() != 0:
                client_flag = 0
                print("Applying train test split . . .")
                X_train, X_test, y_train, y_test = train_test_split(self._cluster_df,
                                                                    self._target,
                                                                    test_size=0.3,
                                                                    stratify=self._target,
                                                                    random_state=self.random_state)
                print("Applying SMOTE oversampling . . .")
                X_train, y_train = SMOTE(n_jobs=-1, random_state=self.random_state).fit_resample(X_train, y_train)
                print("Training Logistic Regression . . .")
                classifier = LogisticRegression(solver="saga",
                                                max_iter=1000,
                                                n_jobs=-1,
                                                class_weight="balanced",
                                                random_state=self.random_state)
                classifier.fit(X_train, y_train)
                print("Making predictions and saving metrics . . .")
                prediction = classifier.predict(X_test)
                train_output.update({cluster: {"client_flag": client_flag,
                                                "classifier": classifier,
                                                "metrics": {"accuracy": accuracy_score(y_test, prediction),
                                                            "precision": precision_score(y_test, prediction),
                                                            "recall": recall_score(y_test, prediction),
                                                            "f1_score": f1_score(y_test, prediction),
                                                            "roc_auc": roc_auc_score(y_test, prediction)}
                                                }
                                      })
            else:
                print("Cluster has no clients, saving {'client_flag': 1} in the output dictionary.")
                client_flag = 1
                train_output.update({cluster: {"client_flag": client_flag}})

            print(169*"-"+"\n")
        self.train_output = train_output # dict output of the training function
        
    def recommend(self, n_recommendations=10, remove_portfolio_ids=True):
        """
        Makes "n_recommendations". Models need to be trained first with method "train_classifiers".
        Use method "train_recommend" to do both steps at once. 
        Recommendations are made for each cluster proportional to the number of clients in them. Recommendations are sorted by their predicted probabilities in descending order.
        :param n_recommendations: integer, default=10, number of recommendations to be made.
        :param remove_portfolio_ids: boolean, default=True, when False IDs from client companies are mantained in the dataset from which recommendations are made. \
When True, IDs from client companies are removed.
        :return recommendations: Pandas DataFrame, contains IDs and predicted probability of recommended clients for portfolio, sorted in descending order by predicted probabilities.
        """
        n_clients = self.rating_df["client"].sum() # total number of clients
        recs_ratio = (self.rating_df.groupby("cluster").sum() / n_clients) # ratio of recommendations per cluster
        recs_per_cluster = round(recs_ratio * n_recommendations, 0) # number of recommendations per cluster
        n_clusters = self.cluster_labels.nunique()[0] # number of clusters
        
        try:
            self.train_output
        except:
            raise Exception("Models haven't been trained. Models need to be trained before making recommendations. Use method 'train_classifiers' or 'train_recommend'")

        recommendations = pd.DataFrame() 

        for cluster in range(n_clusters):
            if self.train_output[cluster]["client_flag"] == 0:
                n_recs = int(recs_per_cluster.iloc[cluster, 0]) # number of recomendations for the cluster in the iteration

                print(f"- Adding {n_recs} recomendations from cluster {cluster} -\n")
                
                if remove_portfolio_ids: # if True, remove companies that are already clients from predictors dataframe (self._cluster_df)
                    self._get_cluster_target_df(self.rating_df[~self.rating_df["client"]], cluster)
                else:
                    self._get_cluster_target_df(self.rating_df, cluster)                
                
                trained_model = self.train_output[cluster]["classifier"]
                proba = pd.Series(trained_model.predict_proba(self._cluster_df)[:, 1]).sort_values(ascending=False, kind="mergesort") # get sorted probabilities
                proba_idx = proba[0:n_recs].index # get indexes for "n_recs" higher probabilities, they map to the companies in the cluster
                cluster_recs = pd.Series(self._cluster_df.iloc[proba_idx, 0].index) # get sorted ids by probability of being client

                cluster_recs = pd.concat([cluster_recs, proba[0:n_recs].reset_index(drop=True)], axis=1, ignore_index=True)
                cluster_recs.columns = ["id", "proba"]

                recommendations = pd.concat([recommendations, cluster_recs], axis=0, ignore_index=True).sort_values(by="proba",
                                                                                                                    kind="mergesort",
                                                                                                                    ascending=False,
                                                                                                                    ignore_index=True)
            else:
                print(f"- Cluster {cluster} has no clients -\n")

            print(169*"-"+"\n")

        return recommendations
    
    def train_recommend(self, n_recommendations=10, remove_portfolio_ids=True):
        """
        Calls method 'train_classifiers' to train logistic regression models for each cluster based on clients portfolio and companies dataset.
        Calls method 'recommend' to use the trained models to recommend leads.
        :param n_recommendations: integer, default=10, number of recommendations to be made.
        :param remove_portfolio_ids: boolean, default=True, when False IDs from client companies are mantained in the dataset from which recommendations are made. \
When True, IDs from client companies are removed.
        :return recommendations: Pandas DataFrame, contains IDs and predicted probability of recommended clients for portfolio, sorted in descending order by predicted probabilities.
        """
        print("\ntrain_recommend -> training . . .\n")

        self.train_classifiers()

        print("\ntrain_recommend -> recommending . . .\n")
        
        recommendations = self.recommend(n_recommendations, remove_portfolio_ids)

        return recommendations