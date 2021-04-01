"""
Code file for finding the important features of the dataset
"""
import pandas
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy

plt.rcParams.update({'font.size': 14})

def univariateSelection(data):
    """
    This returns straight numbers for reflecting importance
    Below, there are sections for normalizing the data if you want to see how that affects your data
    """
    # If we want to MinMaxReduce the data (normalize it)
    ## Uncomment these 4 lines to normalize your data ##
    # columns = data.columns.values[0:len(data.columns.values)]
    # scaler = preprocessing.MinMaxScaler()
    # scaled_df = scaler.fit_transform(data)
    # data = pandas.DataFrame(scaled_df, columns=columns)

    features = data.columns.values[1:len(data.columns.values)]
    X = data.loc[:, features].values  # Separating out the target variables

    ## Put your dependent variable here ##
    y = data.loc[:, ['Ridership']].values  # dependent variable

    # Below, you can change the value of k to return more or less variables for your "best" variables in the dataset
    bestFeatures = SelectKBest(score_func=chi2, k=15)
    fit = bestFeatures.fit(X, y)
    dfScores = pandas.DataFrame(fit.scores_)
    dfColumns = pandas.DataFrame(features)

    # concat two dataframes for better visualization
    featureScores = pandas.concat([dfColumns, dfScores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    # plotfinalScore = pandas.concat([dfColumns, dfScores-min(dfScores)/(max(dfScores)-min(dfScores))], axis=1)
    # plotfinalScore.columns = ['Specs','Score']  # naming the dataframe columns
    # Also, change the 10 below to whatever value you set K to
    print(featureScores.nlargest(17, 'Score'))  # print 10 best features
    # plotfinalScore.plot(kind = 'bar')
    # plt.show()


def featureSelection(data):
    """
    This returns a horizontal bar graph showing the most important features
    """
    # If we want to MinMaxReduce the data (normalize it)
    ## Uncomment these 4 lines to normalize your data ##
    # columns = data.columns.values[0:len(data.columns.values)]
    # scaler = preprocessing.MinMaxScaler()
    # scaled_df = scaler.fit_transform(data)
    # data = pandas.DataFrame(scaled_df, columns=columns)

    features = data.columns.values[1:len(data.columns.values)]
    X = data.loc[:, features].values  # Separating out the target variables

    ## Put your dependent variable here ##
    y = data.loc[:, ['Ridership']].values  # dependent variable

    model = ExtraTreesRegressor(random_state=0)
    model.fit(X, y)
    print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers

    # plot graph of feature importances for better visualization
    feat_importances = pandas.Series(model.feature_importances_, index=features)
    # print(type(feat_importances))
    new = feat_importances.drop(index='Ridership')
    # print(new)
    # Below, change the nlargest() input to however many variables you want showed
    new.nlargest(15).plot(kind='barh')
    plt.title('Feature Selection')
    plt.show()


def correlationHeatmap(data):
    """
    This returns a heatmap of all the variables, showing how each variable correlates with your dependent variable and
        the other variables in the dataset
    """
    # If we want to MinMaxReduce the data (normalize it)
    ## Uncomment these 4 lines to normalize your data ##
    # columns = data.columns.values[0:len(data.columns.values)]
    # scaler = preprocessing.MinMaxScaler()
    # scaled_df = scaler.fit_transform(data)
    # data = pandas.DataFrame(scaled_df, columns=columns)

    corr = data.corr()
    # Drop self-correlations
    dropSelf = numpy.zeros_like(corr)
    dropSelf[numpy.triu_indices_from(dropSelf)] = True
    # Generate color map
    colormap = sns.diverging_palette(220, 10, as_cmap=True)
    # Generate the heatmap, allowing annotations and place floats in the map
    plt.subplots(figsize=(10, 7))
    sns.set(font_scale=2)
    sns.heatmap(corr, cmap=colormap, annot=True, fmt='.1f', annot_kws={"size": 14}, square=False)
    # xticks
    plt.xticks(numpy.arange(len(corr.columns)) + 0.5, corr.columns)
    # yticks
    plt.yticks(numpy.arange(len(corr.columns)) + 0.5, corr.columns)
    plt.tight_layout()
    plt.show()


ridership_data = pandas.read_csv('C:/Users/conno/Desktop/CUIP/CARTA_feature_selection/data/Yatri_Data/Chatt_TAZ_Road_Fields.csv')

ridership_data = ridership_data.fillna(0)

# Drop any variables, if you want
# data = data.drop([], axis=1)
# uncomment which method you want to run
# For these tests, you'll have to define what your dependent variable is
# PCA_testing(data)
univariateSelection(ridership_data)
# featureSelection(block_data)
# correlationHeatmap(data)
# correlationHeatmap(ridership_data)

# univariateSelection(taz_data)
# correlationHeatmap(block_data)
# correlationHeatmap(taz_data)
correlationHeatmap(ridership_data)
# featureSelection(block_data)
# featureSelection(taz_data)
featureSelection(ridership_data)
# correlationHeatmap(block_data)