from Distribution import Distribution
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas
from scipy.stats import gaussian_kde, uniform
from scipy.optimize import brentq
from settings import *
import sklearn.model_selection 
import sklearn.metrics
from sklearn.linear_model import LogisticRegression

if use_multiprocessing:
    import matplotlib
    matplotlib.use('Agg')
    from joblib import Parallel, delayed
    import multiprocessing
    if num_cores is None:
        num_cores = multiprocessing.cpu_count()

## WHITE WINE = 0
## RED WINE   = 1

def read_ds_wine(path, fileName="ds_wine.csv"):
    '''
    Reads the wine dataset and saves it to a Pandas dataframe.
    Also splits the dataset into a dataset of just white wines
    and a dataset of just red wines.
    Returns a list of all features.
    '''

    os.chdir(path)
    ds_wine = pandas.read_csv(fileName, sep=";")
    ds_wine.fillna(" ")

    ds_white = ds_wine[ds_wine['color'] == 0]
    ds_red = ds_wine[ds_wine['color'] == 1]

    features = list(ds_wine.columns[:-2])

    return ds_wine, ds_white, ds_red, features

def intersec(fun1, fun2, lower, upper):
    '''
    Returns the x such that fun1(x) = fun2(x) where x is
    bounded between 'lower' and 'upper'. Used for finding
    a standard classifier between two probability densities.
    '''
    try:
        return brentq(lambda x : fun1(x) - fun2(x), lower, upper)
    except ValueError:
        return None

def logistic_regression(dataSet, showPlot=False, figSize=(10,6)):
    '''
    Applies logistic regression to the given dataset (Code by Piet Daas).
    'showPlot' returns a bar chart of the (nonzero) weights of features.
    It is assumed that the target variable is the last column in the dataset.
    '''

    ##set var
    wv = 200 ##wordvector size
    mindf = 100 ##min document frequency
    WordEmb = False
    char = 3

    ##algotitme selected
    alg = LogisticRegression(penalty='l1', solver='liblinear')
    ##cval = 10 ##Cross validation amount

    ##split in train and test
    ## Draw sample from training set
    y = np.array(dataSet.iloc[:,-1])
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dataSet.loc[:,dataSet.columns[1:-1]], y, test_size=0.20)

    ##alg is selected before loop above
    alg.fit(X_train, y_train)

    ##predict on test set
    ypred = alg.predict(X_test)
    ##get accuracy
    acc = sklearn.metrics.accuracy_score(y_test, ypred, normalize=True)

    # get weights
    # find main contributing features
    weights = alg.coef_[0]
    main_features = [(x,list(dataSet)[y+1]) for (x,y) in zip(sorted(weights), sorted(range(len(weights)), key=lambda k: weights[k])) if x != 0]

    # plot main contributing features
    if showPlot:
        plt.figure(figsize=figSize)
        ax = plt.axes()
        ax.grid(zorder=0)
        plt.xticks(rotation=45)
        plt.title(f"Logistic regression feature weights (Wines dataset)")
        plt.bar([y for (x,y) in main_features],[x for (x,y) in main_features], zorder=3)
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(f'{path}/figures/wine/logres_feature_weights.png', bbox_inches='tight')
        plt.close()

    # save main feature weight results
    f = open(f'{path}/results/wine/logres_feature_weights.txt', 'w')
    f.write(str(main_features))
    f.close()

    return alg, acc, main_features

def kde_1(dataSet, features, mins=None, maxs=None, showPlot=False, figSize=(10,10)):
    '''
    Uses Gaussian Kernel Density Estimation to compute two 1D probability
    densities of a selected feature from the dataset, one for both
    classes (White wines / Red wines).
    Computes the standard classifier as the intersection between the two
    probability densities situated between their maximizing values.
    'features' requires a list of features that will all be estimated separately.
    'mins' and 'maxs' are manually selected bounds for the wine dataset
    which result in clearer plots.
    'showPlot' shows both probability densities for every given feature.
    '''

    classifiers = {}
    for feature in features:
        if None not in (mins, maxs):
            mn = mins[feature]
            mx = maxs[feature]
        else:
            mn = min(dataSet[feature])
            mx = max(dataSet[feature])
        
        # compute KDE's
        eval_points = np.linspace(mn, mx, num=1000)
        f_white = dataSet.loc[dataSet['color'] == 0, feature]
        kde_white = gaussian_kde(f_white)
        y_white = kde_white.pdf(eval_points)
        f_red = dataSet.loc[dataSet['color'] == 1, feature]
        kde_red = gaussian_kde(f_red)
        y_red = kde_red.pdf(eval_points)
    
        # compute standard classifier value
        max_white = eval_points[y_white.argmax()]
        max_red = eval_points[y_red.argmax()]
        c = intersec(kde_white, kde_red, max_white, max_red)

        # manual correction
        if (feature == 'fixed acidity'):
            c = intersec(kde_white, kde_red, 6, 8)

        if c is None:
            print(f"No clear classifier could be found for feature: {feature}.")
        
        # indicates which class (Red/White wine) corresponds to values of the feature below c.
        # as usual, 0 indicates white and 1 indicates red.
        class_underC = int(max_red<max_white)

        classifiers[feature] = (c,class_underC)

        # save plots
        if showPlot:
            plt.figure(figsize=figSize)
            plt.grid()
            plt.plot(eval_points, y_white, color='blue')
            plt.fill_between(eval_points, y_white, color='blue', alpha=0.3)
            plt.plot(eval_points, y_red, color='red')
            plt.fill_between(eval_points, y_red, color='red', alpha=0.3)
            plt.xlim(mn, mx)
            plt.title(f"White wines vs. Red wines distribution - Feature: {feature}")
            blue_patch = mpatches.Patch(color='blue', label='White wine')
            red_patch = mpatches.Patch(color='red', label='Red wine')
            if c is not None:
                plt.axvline(c, c='black')
                black_patch = mpatches.Patch(color='black', label=f'Standard classifier {c:.4f}')
                plt.legend(handles=[blue_patch,red_patch,black_patch])
            else:
                plt.legend(handles=[blue_patch,red_patch])
            plt.savefig(f'{path}/figures/wine/1D_dist_{feature}.png', bbox_inches='tight')
            plt.close()
        
    # save standard classifiers
    f = open(f'{path}/results/wine/classifiers.txt', 'w')
    f.write(str(classifiers))
    f.close()
    
    return kde_white, kde_red, classifiers

def kde_2(dataSet, features, mins=None, maxs=None, showPlot=False, figSize=(10,10)):
    '''
    Uses Gaussian Kernel Density Estimation to compute two 2D probability
    densities of two selected features from the dataset, one for both
    classes (White wines / Red wines).
    'features' requires a pair of features.
    'mins' and 'maxs' are manually selected bounds for the wine dataset
    which result in clearer plots.
    'showPlot' shows both probability densities for every given feature.
    '''
    ds_white = dataSet.loc[dataSet['color'] == 0, features]
    ds_red = dataSet.loc[dataSet['color'] == 1, features]

    if None not in (mins, maxs):
        xmin, ymin = [mins[f] for f in features]
        xmax, ymax = [maxs[f] for f in features]
    else:
        xmin, ymin = [min(dataSet[f]) for f in features]
        xmax, ymax = [max(dataSet[f]) for f in features]
    
    # compute KDE's
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    eval_points = np.vstack([X.ravel(), Y.ravel()])
    x_w, y_w = [ds_white[f] for f in features]
    f_w = np.vstack([x_w, y_w])
    kde_w = gaussian_kde(f_w)
    Z_w = np.reshape(kde_w(eval_points).T, X.shape)

    x_r, y_r = [ds_red[f] for f in features]
    f_r = np.vstack([x_r, y_r])
    kde_r = gaussian_kde(f_r)
    Z_r = np.reshape(kde_r(eval_points).T, X.shape)

    # show plots
    if showPlot:
        fig, ax = plt.subplots(figsize=figSize)
        plt.title(f"White wines vs. Red wines distribution - Features: {features}")
        ax.imshow(np.rot90(Z_w), cmap=plt.cm.Blues,
                extent=[xmin, xmax, ymin, ymax],
                aspect='auto', alpha=0.5)
        ax.imshow(np.rot90(Z_r), cmap=plt.cm.Reds,
                extent=[xmin, xmax, ymin, ymax],
                aspect='auto', alpha=0.5)
        ax.plot(x_w, y_w, 'o', ms=2, c='blue')
        ax.plot(x_r, y_r, 'o', ms=2, c='red')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_xlabel(f"Feature: {features[0]}")
        ax.set_ylabel(f"Feature: {features[1]}")
        blue_patch = mpatches.Patch(color='blue', label='White wine')
        red_patch = mpatches.Patch(color='red', label='Red wine')
        plt.legend(handles=[blue_patch,red_patch])
        plt.savefig(f'{path}/figures/wine/2D_dist_{features}', bbox_inches='tight')
        plt.close()
    return kde_w, kde_r

def kde_3(dataSet, features, mins=None, maxs=None, showPlot=False, createAnimation=False, n_frames=60, figSize=(10,10)):
    '''
    Uses Gaussian Kernel Density Estimation to compute two 3D probability
    densities of three selected features from the dataset, one for both
    classes (White wines / Red wines).
    'features' requires a triple of features.
    'mins' and 'maxs' are manually selected bounds for the wine dataset
    which result in clearer plots.
    'showPlot' shows all samples from the dataset in the 3D space of
    selected features, where red dots indicate red wines, and blue dots
    indicate white wines.
    'createAnimation' creates a 3D animated gif of the above plot spinning
    horizontally, and saves it to the local directory, with 'n_frames' being
    the total number of frames.
    '''

    if showPlot:
        import imageio
        from PIL import Image
        assert n_frames <= 1000, "Number of frames cannot exceed 1000."

    ds_white = dataSet.loc[dataSet['color'] == 0, features]
    ds_red = dataSet.loc[dataSet['color'] == 1, features]

    if None not in (mins, maxs):
        xmin, ymin, zmin = [mins[f] for f in features]
        xmax, ymax, zmax = [maxs[f] for f in features]
    else:
        xmin, ymin, zmin = [min(dataSet[f]) for f in features]
        xmax, ymax, zmax = [max(dataSet[f]) for f in features]
    
    # compute KDE's
    x_w, y_w, z_w = [ds_white[f] for f in features]
    f_w = np.vstack([x_w, y_w, z_w])
    kde_w = gaussian_kde(f_w)
    x_r, y_r, z_r = [ds_red[f] for f in features]
    f_r = np.vstack([x_r, y_r, z_r])
    kde_r = gaussian_kde(f_r)

    # show plot and save animation
    if showPlot:
        plt.figure(figsize=figSize)
        plt.title(f"3D space of features: {features}")
        ax = plt.axes(projection='3d')
        colormap = ['blue']*len(x_w) + ['red']*len(x_r)
        ax.scatter3D(pandas.concat([x_w, x_r]),
                    pandas.concat([y_w, y_r]),
                    pandas.concat([z_w, z_r]), c=colormap)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_zlim([zmin, zmax])
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])
        blue_patch = mpatches.Patch(color='blue', label='White wine')
        red_patch = mpatches.Patch(color='red', label='Red wine')
        plt.legend(handles=[blue_patch,red_patch], loc='upper right')

        if createAnimation:
            frames = []
            frames_dir = f"{path}/figures/wine/frames"
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir)
            for file in os.listdir(frames_dir):
                os.remove(f"{frames_dir}/{file}")
            for i in range(n_frames):
                ax.view_init(elev=20, azim=i*(360/n_frames))
                plt.savefig(f'{frames_dir}/frame_{i:03}.png', bbox_inches='tight')
            for file in os.listdir(frames_dir):
                frames.append(Image.open(f"{frames_dir}/{file}"))
            imageio.mimsave(f'{path}/figures/wine/3D_dist_{features}.gif', frames)
            for file in os.listdir(frames_dir):
                os.remove(f"{frames_dir}/{file}")
            os.rmdir(frames_dir)

    return kde_w, kde_r

def kde_n(dataSet, features):
    '''
    Uses Gaussian Kernel Density Estimation to compute two nD probability
    densities of n selected features from the dataset, one for both
    classes (White wines / Red wines).
    'features' requires a list of features.
    '''

    ds_white = dataSet.loc[dataSet['color'] == 0, features]
    ds_red = dataSet.loc[dataSet['color'] == 1, features]

    x_w = ds_white[features]
    x_r = ds_red[features]

    f_w = np.vstack([x_w[f] for f in features])
    kde_w = gaussian_kde(f_w)
    f_r = np.vstack([x_r[f] for f in features])
    kde_r = gaussian_kde(f_r)

    return kde_w, kde_r

def compute_overlap(dataSet, features, mins=None, maxs=None, samples=20_000):
    '''
    Computes the n-dimensional volume of the overlap between
    two n-dimensional probability densities of n features.
    This is achieved via Monte Carlo integration.
    'samples' is the number of random samples taken for the
    Monte Carlo approach. This should scale exponentially with
    the number of considered features to preserve accuracy.
    '''

    print(f"Computing Monte-Carlo overlap for feature(s): {features}.")
    kernel_w, kernel_r = kde_n(dataSet, features)

    if None not in (mins, maxs):
        mn = [mins[f] for f in features]
        mx = [maxs[f] for f in features]
    else:
        mn = [min(dataSet[f]) for f in features]
        mx = [max(dataSet[f]) for f in features]

    l = [np.linspace(min,max,100) for (min,max) in zip(mn,mx)]
    X = np.meshgrid(*l)

    positions = np.vstack([x.ravel() for x in X])

    zmx_w = max(kernel_w.pdf(positions))
    zmx_r = max(kernel_r.pdf(positions))
    zmx = max(zmx_w, zmx_r)
    below_density_w = 0
    below_density_r = 0
    below_densities = 0
    i = 0

    dist_uniform = [Distribution(uniform(min,max-min)) for (min,max) in zip(mn,mx)]
    dist_uniform_z = Distribution(uniform(0,zmx))

    uniforms = [d.rvs(samples) for d in dist_uniform]
    z_uniforms = dist_uniform_z.rvs(samples)

    while (i < samples):
        x = [u[i] for u in uniforms]
        z = z_uniforms[i]

        z_w = kernel_w.pdf(x)
        z_r = kernel_r.pdf(x)
        if z <= z_w:
            below_density_w += 1
        if z <= z_r:
            below_density_r += 1
        if z <= min(z_w, z_r):
            below_densities += 1
        i += 1

    area_w = ( np.prod([max-min for (min,max) in zip(mn,mx)]) * zmx * below_density_w) / samples
    area_r = ( np.prod([max-min for (min,max) in zip(mn,mx)]) * zmx * below_density_r) / samples
    area_both = ( np.prod([max-min for (min,max) in zip(mn,mx)]) * zmx * below_densities) / samples
    
    return area_w, area_r, area_both, features

def compute_misclassification(dataSet, features, classifiers):
    '''
    Computes the 1D probability of misclassification of a
    randomly selected wine from the dataset from a given feature.
    'features' requires a list of features that will all be considered
    separately.
    'classifiers' is a list of values computed by the method 'kde_1' which
    represent some good values for features to distinguish the two
    classes of wine. These values are used to compute the misclassification
    probabilities for every feature if such a classifier was found.
    '''

    misclassification = {}
    for feature in features:
        c = classifiers[feature]
        if c[0] is None:
            print(f"Misclassification could not be computed for feature: {feature}.")
            misclassification[feature] = None
            continue
        
        kernel_w, kernel_r = kde_n(dataSet, [feature])

        nr_wines = len(ds_wine)
        nr_whites = len(ds_white)
        nr_reds = len(ds_red)
        if c[1] == 0: # feature value under classifier is classified as white
            misclassification[feature] = kernel_r.integrate_box_1d(-400,c[0])*nr_reds/nr_wines + \
                                         kernel_w.integrate_box_1d(c[0], 400)*nr_whites/nr_wines
        if c[1] == 1: # feature value under classifier is classified as red
            misclassification[feature] = kernel_w.integrate_box_1d(-400,c[0])*nr_whites/nr_wines + \
                                         kernel_r.integrate_box_1d(c[0], 400)*nr_reds/nr_wines

    return misclassification

# Manually selected minima and maxima for every feature
# in the wines dataset. These are used both for nicer
# plots and for more accurate results from Monte Carlo
# integration as they reduce the amount of empty space.
mins = {'alcohol': 8,
        'chlorides': 0,
        'citric acid': 0,
        'density': 0.98,
        'fixed acidity': 4,
        'free sulfur dioxide': 0,
        'pH': 2.7,
        'residual sugar': 0,
        'sulphates': 0.2,
        'total sulfur dioxide': 0,
        'volatile acidity': 0}
maxs = {'alcohol': 15,
        'chlorides': 0.2,
        'citric acid': 0.9,
        'density': 1.01,
        'fixed acidity': 15,
        'free sulfur dioxide': 100,
        'pH': 4,
        'residual sugar': 22,
        'sulphates': 1.4,
        'total sulfur dioxide': 280,
        'volatile acidity': 1.3}


# Read wines dataset
print("Reading wines dataset.")
ds_wine, ds_white, ds_red, features = read_ds_wine(f'{path}/data')
if not os.path.exists(f'{path}/figures/wine'):
    os.makedirs(f'{path}/figures/wine')
if not os.path.exists(f'{path}/results/wine'):
    os.makedirs(f'{path}/results/wine')

# Logistic regression, compute weights of features.
# Generate plot of weights in 'figures' folder.
# Save weights locally in 'results' folder.
print("Computing logistic regression.")
alg, acc, main_features = logistic_regression(ds_wine, showPlot=True, figSize=(10,6))


# Compute 1D Guassian KDE's of white/red wine distributions
# for every feature. Generate plots and save the standard
# classifiers where the distributions intersect (if they can be found).
# In the 'classifiers.txt' file, the first value specifies
# its value, the second specifies which color wine corresponds to
# the left of the classifier (0 = white, 1 = red).
print("Computing 1D KDE's.")
kde_white, kde_red, classifiers = kde_1(ds_wine, features, mins, maxs, showPlot=True)


# Compute 2D Gaussian KDE's of white/red wine distributions
# for all combinations of a selection of features. Generate plots.
selected_features = ['density', 'citric acid', 'pH', 'sulphates', 'volatile acidity', 'chlorides']
print("Computing 2D KDE's for all pairs of features in:")
print(selected_features)
combos = combinations(selected_features, 2)
if use_multiprocessing:
    Parallel(n_jobs = num_cores) (delayed(kde_2)(ds_wine, c, mins, maxs, showPlot=True) for c in combos)
if not use_multiprocessing:
    for c in combos:
        kde_2(ds_wine, c, mins, maxs, showPlot=True)


# Compute 3D Gaussian KDE's of white/red wine distributions
# for all combinations of a selection of features. Generate animations.
triples = [('chlorides', 'volatile acidity', 'total sulfur dioxide'),
           ('chlorides', 'volatile acidity', 'sulphates')]
print("Computing 3D KDE's for all triples of features in:")
print(triples)
for c in triples:
    kde_3(ds_wine, c, mins, maxs, showPlot=True, createAnimation=True, n_frames=60)


# Compute 1D overlap of white/red wine distributions
# for every feature. Save the overlap values locally
# to the 'results' folder.
print("Computing overlap values via Monte-Carlo.")
overlaps = {}
if use_multiprocessing:
    res = Parallel(n_jobs = num_cores) (delayed(compute_overlap)(ds_wine, [f], mins, maxs, samples=20_000) for f in features)
    overlaps = {d[0]:c for (a,b,c,d) in res}
if not use_multiprocessing:
    for f in features:
        _,_,overlap,_ = compute_overlap(ds_wine, [f], mins, maxs, samples=20_000)
        overlaps[f] = overlap
f = open(f'{path}/results/wine/overlap.txt', 'w')
f.write(str(overlaps))
f.close()


# Compute 1D misclassification probabilities for
# every feature. Save the misclassification values
# locally to the 'results' folder.
print("Computing misclassification probabilities.")
misclassification = compute_misclassification(ds_wine, features, classifiers)
f = open(f'{path}/results/wine/misclassification.txt', 'w')
f.write(str(misclassification))
f.close()


# Creates a histogram of overlap values and
# misclassification probabilities for every feature.
misclassification_ = {f : (0 if v is None else v) for (f,v) in misclassification.items()}
plt.figure(figsize=(10,6))
ax = plt.axes()
ax.set_axisbelow(True)
plt.xticks(rotation=45)
plt.grid()
plt.title("Overlap and misclassification values for all features")
plt.bar(features, overlaps.values())
plt.bar(features, misclassification_.values())
plt.subplots_adjust(bottom=0.2)
blue_patch = mpatches.Patch(color='blue', label='Area of overlap')
orange_patch = mpatches.Patch(color='orange', label='Probability of misclassification')
plt.legend(handles=[blue_patch,orange_patch])
plt.savefig(f'{path}/figures/wine/overlap_misclassification_values_histogram.png', bbox_inches='tight')
plt.close()