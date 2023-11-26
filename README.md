# Categorization

<p align="justify"> This repository showcases the work our group has done over the course of the modelling week. It presents a variety of methods written in Python meant for visualizing data in 1/2/3 dimensions, approximating sample distributions via Gaussian KDE, logistic regression of features, computing overlap areas and misclassification probabilities for (selections of) features. The final result underlines how overlap areas differ from misclassification probabilities, and how computing overlap can identify features with a better distinction between two classes. </p>

<p align="center">
  <img src="figures/wine/3D_dist_('chlorides', 'volatile acidity', 'sulphates').gif" width="400" height="400"> <img src="figures/wine/3D_dist_('chlorides', 'volatile acidity', 'total sulfur dioxide').gif" width="400" height="400">
</p>

- <p align="justify"> "Wine_categorization.py" is structured to give an automatic showcase of our results. Hopefully it is self-evident how to adapt the methods in this file to your own desire. For more info, see Section **Wine categorization methods** below. </p>

- <p align="justify"> "Website_categorization.py" is structured very similarly to "Wine_categorization.py". Some results will underline that certain methods do not work as well in a large, sparse dataset such as the websites dataset. Given the larger number of Monte Carlo simulations, it might be better to allow multi-processing. This can be done in "settings.py" and requires installing the "joblib" Python package. </p>

## Datasets

We analyze two different datasets:

- <p align="justify"> "Dataset2_TFIDF_anon2.csv" is a large and sparse dataset detailing TF-IDF scores of common terms on company websites. The two classes are innovative and non innovative companies. </p>
- "ds_wine.csv" is a dataset of wines with a restricted feature space detailing qualities of wines. The two classes are red and white wines.

## Settings

 <p align="justify"> Please make sure to change the address of the categorization folder appropriately in "settings.py". Here you can also choose whether to use multiprocessing or not, and specify the number of cores. </p>

## Requirements

 <p align="justify"> In order to properly run the code, some Python packages might need to be installed. See "requirements.txt". If using multiprocessing, be sure to install 'joblib', otherwise this is not needed. </p>

"Distribution.py" is a method written by Marko Boon, professor at the TU/e that speeds up large sampling of random variable distributions.

## Output

Generated figures are saved to the "figures" folder, numerical results are saved to the "results" folder.

## Wine categorization methods

A list of methods in "Wine_categorization.py":

- **read_ds_wine(path, fileName="ds_wine.csv")**:  <p align="justify"> Reads the wine dataset, outputs it to a Pandas dataframe. Includes two separate dataframes for white and red wines, and a list of all features; </p>
- **intersec(fun1, fun2, lower, upper)**:  <p align="justify"> Returns the value where two functions intersect, which is useful for automatically finding a classifier between two probability densities; </p>
- **logistic_regression(dataSet, showPlot=False, figSize=(10,6))**:  <p align="justify"> Code mostly written by Piet Daas. Applies logistic regression to the wine dataset, saves a histogram of feature weights, and returns the algorithm, accuracy, and features with a nonzero weight; </p>
- **kde_1(dataSet, features, mins=None, maxs=None, showPlot=False, figSize=(10,10))**:  <p align="justify"> Uses Gaussian Kernel Density Estimation to compute two 1D probability densities of a selected feature from the dataset, one for both classes (White wines / Red wines).

  Computes the standard classifier as the intersection between the two probability densities. "features" requires a list of features which will all be estimated separately, "mins" and "maxs" are manually selected bounds for feature values.

  Returns plots of both probability densities of white and red wines for each feature. </p>
- **kde_2(dataSet, features, mins=None, maxs=None, showPlot=False, figSize=(10,10))**:  <p align="justify"> Similar to kde_1 but in two dimensions. "features" requires a pair of features. </p>
- **kde_3(dataSet, features, mins=None, maxs=None, showPlot=False, createAnimation=False, n_frames=60, figSize=(10,10))**:  <p align="justify"> Similar to kde_1 but in three dimensions. "features" requires a triple of features.

  Generates a spinning animation of the 3D feature space underlining the two probability densities for white and red wines. </p>
- **kde_n(dataSet, features)**:  <p align="justify"> Generalization of kde to n dimensions without creating figures. </p>
- **compute_overlap(dataSet, features, mins=None, maxs=None, samples=20_000)**:  <p align="justify"> Computes the n-dimensional overlapping area of probability densities of white and red wines for every feature listed in "features". This is done via Monte Carlo, so one should be careful to take enough samples, especially when considering many features simultaneously. </p>
- **compute_misclassification(dataSet, features, classifiers)**:  <p align="justify"> Computes the 1D probability of misclassification of a randomly selected wine from the dataset from a given feature. "features" requires a list of features that will all be considered separately. "classifiers" is the list of classifiers computed previously by the function kde_1. </p>

## Website categorization methods

A list of methods in "Website_categorization.py":

- **read_ds_webs(path, fileName="Dataset2_TFIDF_anon2.csv")**:  <p align="justify"> Reads the websites dataset, deletes columns with only zero values for either class, and outputs it to a Pandas dataframe. Includes two separate dataframes for non-innovative and innovative companies, and a list of all features; </p>
- **intersec(fun1, fun2, lower, upper)**:  <p align="justify"> Returns the value where two functions intersect, which is useful for automatically finding a classifier between two probability densities; </p>
- **logistic_regression(dataSet, showPlot=False, figSize=(10,6))**:  <p align="justify"> Code mostly written by Piet Daas. Applies logistic regression to the websites dataset, saves a histogram of feature weights, and returns the algorithm, accuracy, and features with a nonzero weight; </p>
- **kde_1(dataSet, features, showPlot=False, includeZeroes=False, figSize=(10,10))**:  <p align="justify"> Uses Gaussian Kernel Density Estimation to compute two 1D probability densities of a selected feature from the dataset, one for both classes.

  Computes the standard classifier as the intersection between the two probability densities. "features" requires a list of features which will all be estimated separately, "includeZeroes" indicates whether the (many) zero values should be included.

  Returns plots of both probability densities of non-innovative and innovative companies for each feature. </p>
- **kde_2(dataSet, features, showPlot=False, includeZeroes='none', figSize=(10,10))**:  <p align="justify"> Similar to kde_1 but in two dimensions. "features" requires a pair of features.

  "includeZeroes" eliminates zero values from the features:
  
  - "all" only considers companies with both features nonzero;
  - "any" only considers companies with at least one feature nonzero;
  - "none" does not eliminate zero values. </p>
- **kde_3(dataSet, features, showPlot=False, includeZeroes='none', createAnimation=False, n_frames=60, figSize=(10,10))**:  <p align="justify"> Similar to kde_1 but in three dimensions. "features" requires a triple of features.

  Generates a spinning animation of the 3D feature space underlining the two probability densities for non-innovative and innovative companies. </p>
- **kde_n(dataSet, features, includeZeroes='none')**:  <p align="justify"> Generalization of kde to n dimensions without creating figures. </p>
- **compute_overlap(dataSet, features, includeZeroes='none', samples=20_000)**:  <p align="justify"> Computes the n-dimensional overlapping area of probability densities of non-innovative and innovative companies for every feature listed in "features". This is done via Monte Carlo, so one should be careful to take enough samples, especially when considering many features simultaneously. </p>
- **compute_misclassification(dataSet, features, classifiers, includeZeroes='none')**:  <p align="justify"> Computes the 1D probability of misclassification of a randomly selected website from the dataset from a given feature. "features" requires a list of features that will all be considered separately. "classifiers" is the list of classifiers computed previously by the function kde_1. </p>

## Authors

This project is a collaboration of five students of the TU/e: Pim de Haan, Saskia IJntema, Lieke Janssen, Katarzyna Kasperkiewicz, and Aron van Veldhuizen.
