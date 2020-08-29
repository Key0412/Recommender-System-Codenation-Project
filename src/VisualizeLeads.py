# Related third party imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class VisualizeLeads():
    """
    Class to generate visualizations and tables on the recommended leads.
    """
    def __init__(self, recommended_ids):
        """
        :param recommended_ids: Pandas DataFrame, contains IDs of recommended clients for portfolio, sorted in descending order by predicted probabilities
        """
        self.ids = recommended_ids        
            
    def create_barplots(self, original_market_df, n_labels=5):
        """
        Shows a grid with subplots containing barplots for every feature in the list 'self.report_features'. Counts the frequency of each class for each of the features.
        :param original_market_df: Pandas Dataframe, original dataframe with companies features, \
(e.g. not processed. See more about this in the projects' GitHub page or in the main.ipynb notebook).
        :param n_labels: integer, default=3, representes number of features' labels to plot. Uses the 'n_labels' more frequent features.
        """        
        # Important features from the original dataset, used for visualization/context
        self.report_features = original_market_df.columns
        
        if len(self.report_features) == 1:
            x = original_market_df.loc[self.ids].value_counts().head(n_labels)
            y = x.index        
            plt.figure(figsize = (4, 3))
            sns.barplot(x = x, y = y)
            plt.xlabel(self.report_features[0])
        else:
            n_figures = len(self.report_features) - 1
            nrows = len(self.report_features)
            ncols = 1
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4, nrows*3)) ## ###
            plt.subplots_adjust(hspace=0.3)

            flag = 0
            while flag <= n_figures:
                for pos_row in range(0, nrows):
                    ax = axs[pos_row]
                    x = original_market_df.loc[self.ids, self.report_features[flag]].value_counts().head(n_labels)
                    y = x.index
                    sns.barplot(x=x, y=y, ax=ax, palette="plasma")
                    ax.set_xlabel("")
                    ax.tick_params(labelsize=8)
                    ax.set_title(self.report_features[flag], fontsize=10, fontweight='bold')
                    flag+=1                    
                        
    def create_table(self, original_market_df):
        """
        Create Pandas Dataframe with ranked IDs, showing features of each recommendation.
        :return ranked_table: Pandas Dataframe, shows ranked recommended companies and their features.
        """
        n_recommendations = self.ids.shape[0] 
        ranks = pd.Series([int(rank) for rank in range(1, n_recommendations + 1)], name="Ranking")
        ranked_table = pd.concat([ranks, original_market_df.loc[self.ids].reset_index()], axis=1)
        return ranked_table