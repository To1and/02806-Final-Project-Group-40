from IPython.display import Image, HTML
import json
import datetime
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from wordcloud import WordCloud, STOPWORDS
import plotly
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import chart_studio.tools
import warnings
warnings.filterwarnings('ignore')
chart_studio.tools.set_credentials_file(username='rounakbanik', api_key='xTLaHBy9MVv5szF4Pwan')

sns.set_style('whitegrid')
sns.set(font_scale=1.25)
pd.set_option('display.max_colwidth', 50)

df = pd.read_csv('movies.csv')

df = df.drop('original_title', axis=1)
df = df.drop(['imdb_id'], axis=1)
df = df.drop(['homepage'], axis=1)
df = df.drop(['id'], axis=1)

df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')

df = df.dropna(subset=['revenue', 'budget'])
df = df[(df['revenue'] > 0) & (df['budget'] > 0)]

df.info()

df['return_rate'] = df['revenue'] / df['budget']
df[df['return_rate'].isnull()].shape