# Standard library imports
import base64

# Related third party imports
import pandas as pd

def get_csv_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    :param df:  dataframe
    :return: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<strong><a href="data:file/csv;base64,{b64}" download="recomendacoes.csv">Download de Recomendações</a></strong>'
    return href