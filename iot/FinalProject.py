## USD AAI 530 Analytics and Internet of Things
## Final Project - Module 6/7
## by Bryan Carr
## Submitted 28 Feb 2022

"""
My goal will be to analyse a famous dataset of Spotify songs, found on Kaggle at:
https://www.kaggle.com/mrmorj/dataset-of-songs-in-spotify

I aim to predict the genre based on the input variables.
To simplify, I will focus on only one genre at a time - predicting if a song is part of a genre or not.
I will use simple Decision Tree Classifiers, and more advanced Gradient Boosted Decision Tree Classifiers, noting the difference.

Later I will add a visual wrapper to input several paramaters from the user

I see several functions as required:
1) Main Function - Reads in dataset and calls other functions
2) EDA
3) Decision Tree Classifier
4) Gradient Boosted Tree Classifier
5) Visual Input

"""

## Import the required libraries
import pandas
import pandas as pd, numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, f1_score
import xgboost as xgb
import streamlit as st

#Main Function: runs the project & sub-functions
#Requires Spotify Data in the directory
#Inputs: Log Text file name
#    tree_paramaters: a dictionary with max_Depth; min_samples_split; and min_impurity_decrease
#    xgb_paramaters: a dictionary with max_depth; num_steps; learning_rate; and gamma
#Outputs a Text log file with results
def main_function(log_file = "", tree_paramaters = {}, xgb_paramaters = {}):
    # Read in Data
    spotify_df = pd.read_csv("genres_v2.csv")

    # Drop unhelpful Object columns
    spotify_df = spotify_df.drop(
        columns=['type', 'id', 'uri', 'track_href', 'analysis_url', 'song_name', 'Unnamed: 0', 'title'])

    # Replace similar genres:
    # Under Ground Rap => Rap
    # Dark Trap => Trap
    # Trap Metal => Trap
    # Tech house => Techno
    # Psytrance => Trance
    # DnB and Hardstyle seem to be similar but different - I won't touch them
    spotify_df.replace('Dark Trap', 'trap', inplace=True)
    spotify_df.replace('Underground Rap', 'Rap', inplace=True)
    spotify_df.replace('Trap Metal', 'trap', inplace=True)
    spotify_df.replace('techhouse', 'techno', inplace=True)
    spotify_df.replace('psytrance', 'trance', inplace=True)

    # Do EDA:
    exploratory_data_analysis(spotify_df)

    # Create list of Genres to Iterate over
    genre_list = spotify_df['genre'].unique().tolist()


    # Create Decision Tree Classifier paramaters
#    tree_paramaters = {
#        'max_depth' : [10,15,20,30],
#        'min_samples_split' : [2,5,10],
#        'min_impurity_decrease' : [0.0,0.001,0.01]
#    }


    # Write Tree Params to log file
    with open(str(log_file), 'a') as f:
        f.write("******************************\n")
        f.write("******************************\n")
        f.write("Results from simple Decision Tree Classifiers, with 4-fold Cross Validation over the following Parameters" + '\n')
        f.write(str(tree_paramaters) + "\n")
        f.write("******************************\n")
        f.write("******************************\n")

    # Initialize a Dictionary of Scores - this will track our XGB progress
    tree_scores_dict = {'Genre' : [], 'Accuracy' : [], 'F1' : []}

    # Iterate through all Decision Tree Classifiers
    for g in genre_list:
        #Format Data for this Genre
        formated_data = filter_genres(spotify_df, g)

        # Log results - most results to be logged in ML function
        with open(str(log_file), 'a') as f:
            f.write("====================\n")
            f.write("Tree Results for Genre: " + str(g) + '\n')
            f.write("====================\n")

        # Run the Tree Classifier CV
        tree_acc, tree_f1 = grad_boost_classifier(tree_paramaters, formated_data, log_file)

        # Append scores to the Dictionary
        tree_scores_dict['Genre'] += [str(g)]
        tree_scores_dict['Accuracy'] += [float(tree_acc)]
        tree_scores_dict['F1'] += [float(tree_f1)]
    #End of Loop

    # Convert Scores Dict to Dataframe for logging & plotting
    tree_scores_df = pandas.DataFrame(tree_scores_dict)
    print("XGB Scores Dataframe:")
    print(tree_scores_df)

    # Write Scores DF to the Results text:
    with open(str(log_file), 'a') as f:
        f.write("====================\n")
        f.write("Dataframe of Tree Results: " + '\n')
        f.write(tree_scores_df.to_string() + "\n\n\n")

    # Melt/Expand Scores DF for plotting purposes
    tree_scores_df_long = tree_scores_df.melt(id_vars=['Genre'],
                                            value_vars=['Accuracy', 'F1'],
                                            var_name='Score Type',
                                            value_name='Score')
    # Plot Scores as a Bar Plot: easy to visualize
    sns.barplot(data=tree_scores_df_long, x='Genre', y='Score', hue='Score Type').set(
        title="Accuracy and F1 for Best Decision Trees")
    plt.xticks(rotation=45)
    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()

    # Create XGBoost Tree Classifier Paramaters
#    xgb_paramaters = {
#        "max_depth": [2,3,4,5,6],
#        "learning_rate": [0.1,0.2,0.3],
#        "num_steps": [10,15,20],
#        "gamma": [0.1,0.3,0.5]
#    }

    # Write XGB Parameters to the Log File
    with open(str(log_file), 'a') as f:
        f.write("******************************\n")
        f.write("******************************\n")
        f.write("Results from XGB, with 4-fold Cross Validation over the following Parameters" + '\n')
        f.write(str(xgb_paramaters) + "\n")
        f.write("******************************\n")
        f.write("******************************\n")

    # Initialize a Dictionary of Scores - this will track our XGB progress
    xgb_scores_dict = {'Genre' : [], 'Accuracy' : [], 'F1' : []}

    # Loop over Genres, running XGB for each
    for g in genre_list:
        #Format Data for this Genre
        formated_data = filter_genres(spotify_df, g)

        # Initialize Log for results - Results will be logged in ML function called below
        with open(str(log_file), 'a') as f:
            f.write("====================\n")
            f.write("XGB Results for Genre: " + str(g) + '\n')
            f.write("====================\n")

        # Run XGB Test & track scores in a dataframe
        acc, f1 = grad_boost_classifier(xgb_paramaters, formated_data, log_file)
        print("Above XGB results are for target genre: " + str(g))
        print("Accuracy: " + str(acc) + "  and F1 Score: " + str(f1))

        # Append scores to the Dictionary
        xgb_scores_dict['Genre'] += [str(g)]
        xgb_scores_dict['Accuracy'] += [float(acc)]
        xgb_scores_dict['F1'] += [float(f1)]
    #End of Loop

    # Convert Scores Dict to Dataframe for logging & plotting
    xgb_scores_df = pandas.DataFrame(xgb_scores_dict)
    print("XGB Scores Dataframe:")
    print(xgb_scores_df)

    # Write Scores DF to the Results text:
    with open(str(log_file), 'a') as f:
        f.write("====================\n")
        f.write("Dataframe of Results: " + '\n')
        f.write(xgb_scores_df.to_string() + "\n")

    # Melt/Expand Scores DF for plotting purposes
    xgb_scores_df_long = xgb_scores_df.melt(id_vars=['Genre'],
                                    value_vars=['Accuracy', 'F1'],
                                    var_name='Score Type',
                                    value_name='Score')
    # Plot Scores as a Bar Plot: easy to visualize
    sns.barplot(data=xgb_scores_df_long, x='Genre', y='Score', hue='Score Type').set(
        title="Accuracy and F1 for Best XGB Models")
    plt.xticks(rotation=45)
    #plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.show()

    print("All done Main Function!")
    return()

#EDA Function
#Inputs: Dataframe to perform EDA on
#Outputs: Nil, but prints Graphs
#Creates Box plots and Histograms for all variables in Dataset
def exploratory_data_analysis(dataset):
    print("Dataset Imported, with Info:")
    print(dataset.info())

    # Create a boxplot for all Numerical variables ranging from 0 to 1
    sns.boxplot(data=dataset.drop(['duration_ms', 'loudness', 'tempo', 'key', 'time_signature'], axis=1), orient='h')
    plt.show()

    # Plot the other variables' boxplots separately
    for variable in ['duration_ms', 'loudness', 'tempo', 'key', 'time_signature']:
        sns.boxplot(data=dataset.loc[:, variable], orient='h').set(title="Boxplot for " + str(variable))
        plt.show()

    # Plot histograms for all numeric variables
    spotify_histograms = dataset.hist(layout=(5, 3), figsize=(15, 25))
    plt.show()
    return()

#Filter Genre Data function
#Input: Dataframe, and string for Target Variable to save
#Output: Dataframe with Genre info reformatted as 'In Target' or 'Not in Target', pseudo-Binary
#Make a new column showing if the Song is in our Target Genre (True or 1.0) or not (False or 0.0)
#Then drop the old Genre column
def filter_genres(dataset, target):

    in_target = []

    for i in dataset['genre']:
        if i == target:
            in_target.append(1.0)
        else:
            in_target.append(0.0)

    dataset['in_genre'] = in_target
    dataset = dataset.drop('genre', axis=1)
    return dataset

#XGB Classifier
#Takes in: Paramaters, Dataframes
#Outputs: model's Accuracy and F1 scores, for evaluation

#Creates an XGB Classifier with the input Params
#Splits dataframe into Test and Train
#Runs XGB Classifier
#Prints scores and Confusion matrix
#Returns Scores
def grad_boost_classifier(params, dataset, LogFile = ""):
    # Build Classifier model
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', booster='gbtree', use_label_encoder=False,
                                  random_state=1234, eval_metric='error')

    # Split Data: Build Test-Train Splits, with a Test Size of 20%
    X_train, X_test, Y_train, Y_test = train_test_split(dataset.drop(columns='in_genre'),
                                                        dataset['in_genre'], test_size=0.2,
                                                        random_state=1234)

    # Cross Validation with 4 folds
    grid_search = GridSearchCV(xgb_model, params, n_jobs=-1, cv=4, verbose=2)
    grid_xgb_class = grid_search.fit(X_train, Y_train)

    print("Best Paramaters for XGB: ")
    print(grid_xgb_class.best_params_)
    print("")

    accuracy = grid_xgb_class.best_score_
    print("Best Accuracy score for the best XGB Model: " + str(accuracy))

    # Predict and print Confusion Matrix and F1 Score
    predicted_xgb = grid_xgb_class.predict(X_test)
    print("Confusion Matrix for Best XGB Model: ")
    print(confusion_matrix(Y_test, predicted_xgb))
    print("")
    # Score results
    f1 = f1_score(Y_test, predicted_xgb)
    print("F1 Score for Most Accurate XGB Model: " + str(f1))
    print("")
    print("All done with XGB Classifier!")

    # Write results to log file:
    with open(str(LogFile), 'a') as f:
        f.write("Best Model's Parameters: \n")
        f.write(str(grid_xgb_class.best_params_))
        f.write("\n")
        f.write("Best Accuracy: " + str(accuracy) + "\n")
        f.write("Best F1: " + str(f1) + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(confusion_matrix(Y_test, predicted_xgb)))
        f.write("\n\n")

    return(accuracy, f1)


#Decision Tree Classifier function
#Input: Tree Paramaters (dictionary); Dataframe; LogFile's name (string)
#Outputs: Accuracy and F1 Scores
def tree_classifier(params, dataset, LogFile = ""):
    # Build Decision Tree Classifier
    tree_model = DecisionTreeClassifier(params, random_state=1234)

    # Split Data: Build Test-Train Splits, with a Test Size of 20%
    X_train, X_test, Y_train, Y_test = train_test_split(dataset.drop(columns='in_genre'),
                                                        dataset['in_genre'], test_size=0.2,
                                                        random_state=1234)

    # Cross Validation with 4 folds
    grid_search = GridSearchCV(tree_model, params, n_jobs=-1, cv=4, verbose=2)
    grid_tree_class = grid_search.fit(X_train, Y_train)

    # Score for Accuracy
    accuracy = grid_tree_class.best_score_

    # Predict and print Confusion Matrix and F1 Score
    predicted_tree = grid_tree_class.predict(X_test)

    # Score results
    f1 = f1_score(Y_test, predicted_tree)

    # Write results to log file:
    with open(str(LogFile), 'a') as f:
        f.write("Best Tree Model's Parameters: \n")
        f.write(str(grid_tree_class.best_params_))
        f.write("\n")
        f.write("Best Accuracy: " + str(accuracy) + "\n")
        f.write("Best F1: " + str(f1) + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(confusion_matrix(Y_test, predicted_tree)))
        f.write("\n\n")

    return(accuracy, f1)

# Run the Main Function
#main_function('results.txt')
