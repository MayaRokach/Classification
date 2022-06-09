import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline
from sklearn.cluster import KMeans
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
#Import models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC
#Model evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score,f1_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report


data = pd.read_csv(r'C:\Users\Maya\school\second year\semester B\Data Minning\music_proj\data\spotify_data_genre_cleaned.csv')

data = data[data.genre != 'A Capella']

Z = data[['genre','acousticness','danceability','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence']]
CHARACTERISTICS = ['acousticness','danceability','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence']
GENRES = data["genre"].unique().tolist()
print(GENRES)

def char_differneces_between_genres(genres, characteristics):
    """
    This functions genrates bar plot for each genre showing genres average characteristics.
    :param genres list
    :param characteristics list
    :return: 27 barplots for 27 different genres.
    """
    for genre in genres:
        plt.rcParams['font.size'] = '5'
        X_filtered = Z.query(f"genre == '{genre}'")
        means_list = []
        for character in characteristics:
            X_mean = X_filtered[character].mean()
            means_list.append(X_mean)
        means_list[-2] = means_list[-2]/100
        means_list[-4] = means_list[-4] / 10
        y_pos = np.arange(len(characteristics))
        plt.bar(y_pos, means_list, align='center', width=0.8, color=(0.2, 0.4, 0.6, 0.6))
        # Create names on the x-axis
        plt.xticks(y_pos, characteristics)
        plt.title(f"{genre} characteristic distribution")
        plt.xlabel('characteristics')
        plt.ylabel('mean values')
        # Show graphic
        plt.show()


def create_hist():
    bar_cols = data[['genre','key','mode','time_signature']].columns.values
    for col in bar_cols:
        df_temp = data.groupby([col]).size().reset_index(name='count')
        print(df_temp)
        print(df_temp)
        plt.figure(figsize=(18,8))
        plt.xticks(rotation=45)
        sns.set_style("ticks")
        sns.barplot(data = df_temp, x= col, y= 'count')
        plt.show()

def col_correlations():
    plt.figure(figsize=(10,8))
    sns.heatmap(data.corr(),annot=True)
    plt.show()


def data_description():
    description_table = data.describe()
    print(description_table)

def drop_irrelevant_features():
    """
    this function drop the irrelevant features for genre classificatiion
    :return: dataframe without columns: 'artist name', 'track name', 'track id'
    """
    unused_col = ['artist_name', 'track_name', 'track_id', 'time_signature']
    df = data.drop(columns=unused_col).reset_index(drop=True)
    return df

relevant_data = drop_irrelevant_features()

def convert_to_numeric_only(df):
    mode_dict = {'Major': 1, 'Minor': 0}
    key_dict = {'C': 1, 'C#': 2, 'D': 3, 'D#': 4, 'E': 5, 'F': 6,
                'F#': 7, 'G': 9, 'G#': 10, 'A': 11, 'A#': 12, 'B': 12} # maybe entrances with no key
    genre_dict = {'Movie': 1, 'R&B': 2, 'A Capella': 3, 'Alternative': 4, 'Country': 5, 'Dance': 6, 'Electronic': 7, 'Anime': 8, 'Folk': 9, 'Blues': 10, 'Opera': 11, 'Hip-Hop': 12, "Children's Music": 13, 'Rap': 14, 'Indie': 15, 'Classical': 16, 'Pop': 17, 'Reggae': 18, 'Reggaeton': 19, 'Jazz': 20, 'Rock': 21, 'Ska': 22, 'Comedy': 23, 'Soul': 24, 'Soundtrack': 25, 'World': 26}
    # time_signature_dict = {'4/4': 4, '5/4': 5, '6/4': 6, '7/4': 7}

    # df['time_signature'] = df['time_signature'].replace(time_signature_dict)
    df['mode'] = df['mode'].replace(mode_dict).astype(int)
    df['key'] = df['key'].replace(key_dict).astype(int)
    df['genre'] = df['genre'].replace(genre_dict).astype(int)
    return df

df = convert_to_numeric_only(relevant_data)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.reset_index()



X = df.iloc[:, 3:]
y = df.iloc[:, 2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# models={"LogReg":LogisticRegression(),
#        "KNN":KNeighborsClassifier(),
#        "Random Forest":RandomForestClassifier(),
#         "SVC": SVC()}

# def fit_and_score (models,X_train,X_test,y_train,y_test):
#     """
#     Fits and evaluates given machine learning models
#     """
#     np.random.seed(1)
#     model_scores={}
#     for name , model in models.items():
#         model.fit(X_train,y_train)
#         model_scores[name]=model.score(X_test,y_test)
#     return model_scores


#fit the logistic regression to the data
LR_model = LogisticRegression().fit(X_train,y_train)

#print the coefficient of determination:
LR_model_score = LR_model.score(X_test, y_test)
print('coefficient of Logistic Regression determination is: ', LR_model_score) #before erasing A capella the result was: 0.12544849070791708, now: 0.09300638421358097


KNN_model = KNeighborsClassifier().fit(X_train, y_train)
KNN_model_score = KNN_model.score(X_test, y_test)
print('coefficient of KNN determination is: ', KNN_model_score) #result is 0.33316674190150686

RF_model = RandomForestClassifier().fit(X_train, y_train)
RF_model_score = RF_model.score(X_test, y_test)
print('coefficient of Random Forest determination is: ', RF_model_score) #result is 0.8477891704821479!!!!!!!!

# SVC_model = SVC().fit(X_train, y_train)
# SVC_model_score = SVC_model.score(X_train, y_train)
# print('coefficient of SVC determination is: ', SVC_model_score)