from features import qualityFeatures,relativeAggregation,goalScoredFeatures
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# Aggregates quality features (e.g. Accurate/Inaccurate simple passes, Headers, Lanuches, Assists, Dangerous lost balls...)
qualityFeat = qualityFeatures.qualityFeatures()
data_folder_path = "your\folder\path\to\figshare_downloads\from\'A Dataset for football, Cintia-Pappalardo'\\"
quality=qualityFeat.createFeature(events_path = data_folder_path+'data/events/*.json',
                        players_file=data_folder_path+'data/players.json', entity = 'team')
# Aggregates the feature 'goal-scored' defined as the difference between goal scored and conceded ina a game
gs=goalScoredFeatures.goalScoredFeatures()
goals=gs.createFeature(data_folder_path+'data/matches/*.json')

# Merging of quality features and goals scored
aggregation = relativeAggregation.relativeAggregation()
aggregation.set_features([quality,goals])
df = aggregation.aggregate(to_dataframe = True)
df["goal-scored"] = df.pop("goal-scored") # brings the goal-score column at the end

class FeaturesPreprocessing():
    """
    Aggregated data preprocessing.

    Parameters
    ----------
        dataframe : pandas DataFrame
            a dataframe containing the feature values and the target values

        target: str
            a string indicating the name of the target variable in the dataframe
    """
    def __init__(self, dataframe, target, var_threshold = 0.02):
        ##feature selection by variance, to delete outlier features
        feature_names = list(dataframe.columns)
        # eliminate the variables with zero variance
        sel = VarianceThreshold(var_threshold)
        X = sel.fit_transform(dataframe)
        # print the filtered out variables
        selected_feature_names = [feature_names[i] for i, var in enumerate(list(sel.variances_)) if var > var_threshold]
        filtered_features = [(feature_names[i],var) for i, var in enumerate(list(sel.variances_)) if var <= var_threshold]
        print ("[Weighter] filtered features:", filtered_features, len(filtered_features))
        # create the final dataframe
        self.dataframe = pd.DataFrame(X, columns=selected_feature_names)
        # create the labels Series
        y = self.dataframe[target].apply(lambda x: 1 if x > 0 else -1)
        # Add the label column to the dataframe and rename it ('goal-scored' disappears)
        self.dataframe[target] = y
        self.dataframe.rename({target: 'victory-defeat'}, axis=1, inplace=True)
    def return_dataframe(self):
        return self.dataframe

fetch_data = FeaturesPreprocessing(dataframe = df, target = "goal-scored", var_threshold = 0.02)
# fetch_data.return_dataframe().to_csv("aggregated_data.csv")
