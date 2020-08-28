# Related third party imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class VisualizeLeads():
    """
    Class to generate visualizations and tables on the recommended leads.
    """
    def __init__(self, recommended_ids, original_market_df, report_features=None):
        """
        Define from which companies visualizations are to be created. Visualizations are based on the 'original market dataframe' (estaticos_market.csv).
        :param recommended_ids: Pandas DataFrame, contains IDs of recommended clients for portfolio, sorted in descending order by predicted probabilities
        :param original_market_df: Pandas Dataframe, original dataframe with companies features, \
(e.g. not processed. See more about this in the projects' GitHub page or in the main.ipynb notebook).
        :param report_features: list of strings, default=None, has to contain names of features to be retrieved the original dataframe.\
If not defined, the features id, de_natureza_juridica, sg_uf, de_ramo, setor, idade_emp_cat, de_nivel_atividade, and de_faixa_faturamento_estimado are retrieved.
        """
        self.ids = recommended_ids        
        # Important features from the original dataset, used for visualization/context
        if report_features == None:
            self.report_features = "de_natureza_juridica sg_uf de_ramo setor idade_emp_cat de_faixa_faturamento_estimado".split()
            print(f"Features to visualize from original dataframe: {self.report_features}")
        else:
            self.report_features = report_features
            print(f"Features to visualize from original dataframe: {self.report_features}")
        
        try:
            self.df = original_market_df.loc[self.ids, self.report_features]
        except:
            raise Exception("Check your features again, are you sure they're present on the original dataset?")        
            
    def create_barplots(self, n_labels=10):
        """
        Shows a grid with subplots containing barplots for every feature in the list 'self.report_features'. Counts the frequency of each class for each of the features.
        :param n_labels: integer, default=3, representes number of features' labels to plot. Uses the 'n_labels' more frequent features.
        """    
        if len(self.report_features) == 1:
            x = self.df.value_counts().head(n_labels)
            y = x.index        
            plt.figure(figsize = (20, 10))
            sns.barplot(x = x, y = y)
            plt.xlabel(self.report_features[0])
        else:
            n_figures = len(self.report_features) - 1
            nrows = len(self.report_features)
            ncols = 1
            fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, nrows*12))
            plt.subplots_adjust(hspace=0.4)

            flag = 0
            while flag <= n_figures:
                for pos_row in range(0, nrows):
                    ax = axs[pos_row]
                    x = self.df[self.report_features[flag]].value_counts().head(n_labels)
                    y = x.index
                    sns.barplot(x=x, y=y, ax=ax, palette="plasma")
                    ax.set_xlabel("")
                    ax.tick_params(labelsize=25)
                    ax.set_title(self.report_features[flag], fontsize=25, pad=12, fontweight='bold')
                    flag+=1                    
                        
    def create_table(self):
        """
        Create Pandas Dataframe with ranked IDs, showing features of each recommendation.
        :return ranked_table: Pandas Dataframe, shows ranked recommended companies and their features.
        """
        n_recommendations = self.df.shape[0] 
        ranks = pd.Series([int(rank) for rank in range(1, n_recommendations + 1)], name="Ranking")
        ranked_table = pd.concat([ranks, self.df.reset_index()], axis=1)
        return ranked_table