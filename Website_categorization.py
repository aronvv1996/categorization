from Distribution import Distribution
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas
import random
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

## NON-INNOVATIVE   = 0
## INNOVATIVE       = 1

def read_ds_webs(path, fileName="Dataset2_TFIDF_anon2.csv"):
    '''
    Reads the websites dataset and saves it to a Pandas dataframe.
    Also splits the dataset into a dataset of just non-innovative
    companies and just innovative companies.
    Returns a list of all features.
    '''

    os.chdir(path)
    ds_webs = pandas.read_csv(fileName, sep=";")
    ds_webs.fillna(" ")

    # delete all zero columns
    ds_webs = ds_webs.loc[:, (ds_webs != 0).any(axis=0)]
    # delete 'f879' feature
    ds_webs = ds_webs.drop(['f879'], axis=1)

    ds_ninnov = ds_webs[ds_webs['Innov'] == 0]
    ds_innov = ds_webs[ds_webs['Innov'] == 1]

    # delete all columns with only zero values for non-innovative companies
    ds_webs = ds_webs.loc[:, (ds_ninnov != 0).any(axis=0)].join(ds_webs['Innov'])
    # delete all columns with only zero values for innovative companies
    ds_webs = ds_webs.loc[:, (ds_innov != 0).any(axis=0)]

    features = list(ds_webs.columns[1:-1])

    return ds_webs, ds_ninnov, ds_innov, features

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
        plt.xticks(rotation=90)
        plt.title(f"Logistic regression feature weights (websites dataset)")
        plt.bar([y for (x,y) in main_features],[x for (x,y) in main_features], zorder=3)
        plt.subplots_adjust(bottom=0.2)
        plt.savefig(f'{path}/figures/webs/logres_feature_weights.png', bbox_inches='tight')
        plt.close()

    # save main feature weight results
    f = open(f'{path}/results/webs/logres_feature_weights.txt', 'w')
    f.write(str(main_features))
    f.close()

    return alg, acc, main_features

def kde_1(dataSet, features, showPlot=False, includeZeroes=False, figSize=(10,10)):
    '''
    Uses Gaussian Kernel Density Estimation to compute two 1D probability
    densities of a selected feature from the dataset, one for both classes.
    Computes the standard classifier as the intersection between the two
    probability densities situated between their maximizing values.
    'features' requires a list of features that will all be estimated separately.
    'showPlot' shows both probability densities for every given feature.
    'includeZeroes' eliminates all zero values when creating plots.
    '''

    classifiers = {}
    for feature in features:
        mn = min(dataSet[feature])
        mx = max(dataSet[feature])
        
        # compute KDE's
        eval_points = np.linspace(mn, mx, num=1000)
        f_ninnov = dataSet.loc[dataSet['Innov'] == 0, feature]
        f_innov = dataSet.loc[dataSet['Innov'] == 1, feature]
        if not includeZeroes:
            f_ninnov = f_ninnov.loc[lambda x : x>0]
            f_innov = f_innov.loc[lambda x : x>0]
        kde_ninnov = gaussian_kde(f_ninnov)
        y_ninnov = kde_ninnov.pdf(eval_points)
        kde_innov = gaussian_kde(f_innov)
        y_innov = kde_innov.pdf(eval_points)
    
        # compute standard classifier value
        max_ninnov = eval_points[y_ninnov.argmax()]
        max_innov = eval_points[y_innov.argmax()]
        c = intersec(kde_ninnov, kde_innov, min(max_ninnov, max_innov), mx)

        if c is None:
            print(f"No clear classifier could be found for feature: {feature}.")
        
        # indicates which class corresponds to values of the feature below c.
        # as usual, 0 indicates non-innovative and 1 indicates innovative.
        class_underC = int(max_innov<max_ninnov)

        classifiers[feature] = (c,class_underC)

        # save plots
        if showPlot:
            plt.figure(figsize=figSize)
            plt.grid()
            plt.plot(eval_points, y_ninnov, color='blue')
            plt.fill_between(eval_points, y_ninnov, color='blue', alpha=0.3)
            plt.plot(eval_points, y_innov, color='red')
            plt.fill_between(eval_points, y_innov, color='red', alpha=0.3)
            plt.xlim(mn, mx)
            plt.title(f"Non-innovative vs. Innovative companies distribution - Feature: {feature}")
            blue_patch = mpatches.Patch(color='blue', label='Non-innovative')
            red_patch = mpatches.Patch(color='red', label='Innovative')
            if c is not None:
                plt.axvline(c, c='black')
                black_patch = mpatches.Patch(color='black', label=f'Standard classifier {c:.4f}')
                plt.legend(handles=[blue_patch,red_patch,black_patch])
            else:
                plt.legend(handles=[blue_patch,red_patch])
            plt.savefig(f'{path}/figures/webs/1D_dist_{feature}.png', bbox_inches='tight')
            plt.close()
        
    # save standard classifiers
    f = open(f'{path}/results/webs/classifiers.txt', 'w')
    f.write(str(classifiers))
    f.close()
    
    return kde_ninnov, kde_innov, classifiers

def kde_2(dataSet, features, showPlot=False, includeZeroes='none', figSize=(10,10)):
    '''
    Uses Gaussian Kernel Density Estimation to compute two 2D probability
    densities of two selected features from the dataset, one for both
    classes.
    'features' requires a pair of features.
    'showPlot' shows both probability densities for every given feature.
    'includeZeroes' eliminates zero values:
        - 'all' only considers companies with both features nonzero;
        - 'any' only considers companies with at least one feature nonzero;
        - 'none' does not eliminate zero values.
    '''

    ds_ninnov = dataSet.loc[dataSet['Innov'] == 0, features]
    ds_innov = dataSet.loc[dataSet['Innov'] == 1, features]
    if includeZeroes == 'all':
        ds_ninnov = ds_ninnov[(ds_ninnov !=0).all(axis=1)]
        ds_innov = ds_innov[(ds_innov !=0).all(axis=1)]
    if includeZeroes == 'any':
        ds_ninnov = ds_ninnov[(ds_ninnov !=0).any(axis=1)]
        ds_innov = ds_innov[(ds_innov !=0).any(axis=1)]
    if (len(ds_ninnov) < 3 or len(ds_innov) < 3):
        print(f"Not enough nonzero samples exist for features {features}. Try setting 'includeZeroes' to None or any.")
        return

    xmin, ymin = [min(dataSet[f]) for f in features]
    xmax, ymax = [max(dataSet[f]) for f in features]
    
    # compute KDE's
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    eval_points = np.vstack([X.ravel(), Y.ravel()])
    x_ninnov, y_ninnov = [ds_ninnov[f] for f in features]
    f_ninnov = np.vstack([x_ninnov, y_ninnov])
    x_innov, y_innov = [ds_innov[f] for f in features]
    f_innov = np.vstack([x_innov, y_innov])
    try:
        kde_ninnov = gaussian_kde(f_ninnov)
        kde_innov = gaussian_kde(f_innov)
    except np.linalg.LinAlgError:
        print(f"Not enough nonzero samples exist for features {features}. Try setting 'includeZeroes' to None or any.")
        return
    Z_ninnov = np.reshape(kde_ninnov(eval_points).T, X.shape)
    Z_innov = np.reshape(kde_innov(eval_points).T, X.shape)

    # show plots
    if showPlot:
        fig, ax = plt.subplots(figsize=figSize)
        plt.title(f"Non-innovative vs. Innovative companies distribution - Features: {features}")
        ax.imshow(np.rot90(Z_ninnov), cmap=plt.cm.Blues,
                extent=[xmin, xmax, ymin, ymax],
                aspect='auto', alpha=0.8)
        ax.imshow(np.rot90(Z_innov), cmap=plt.cm.Reds,
                extent=[xmin, xmax, ymin, ymax],
                aspect='auto', alpha=0.5)
        ax.plot(x_ninnov, y_ninnov, 'o', ms=2, c='blue')
        ax.plot(x_innov, y_innov, 'o', ms=2, c='red')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_xlabel(f"Feature: {features[0]}")
        ax.set_ylabel(f"Feature: {features[1]}")
        blue_patch = mpatches.Patch(color='blue', label='Non-innovative')
        red_patch = mpatches.Patch(color='red', label='Innovative')
        plt.legend(handles=[blue_patch,red_patch])
        plt.savefig(f'{path}/figures/webs/2D_dist_{features}', bbox_inches='tight')
        plt.close()
    return kde_ninnov, kde_innov

def kde_3(dataSet, features, showPlot=False, includeZeroes='none', createAnimation=False, n_frames=60, figSize=(10,10)):
    '''
    Uses Gaussian Kernel Density Estimation to compute two 3D probability
    densities of three selected features from the dataset, one for both
    classes.
    'features' requires a triple of features.
    'showPlot' shows all samples from the dataset in the 3D space of
    selected features, where red dots indicate innovative websites, and
    blue dots indicate non-innovative websites
    'includeZeroes' eliminates zero values:
        - 'all' only considers companies with both features nonzero;
        - 'any' only considers companies with at least one feature nonzero;
        - 'none' does not eliminate zero values.
    'createAnimation' creates a 3D animated gif of the above plot spinning
    horizontally, and saves it to the local directory, with 'n_frames' being
    the total number of frames.
    '''

    if showPlot:
        import imageio
        from PIL import Image
        assert n_frames <= 1000, "Number of frames cannot exceed 1000."

    ds_ninnov = dataSet.loc[dataSet['Innov'] == 0, features]
    ds_innov = dataSet.loc[dataSet['Innov'] == 1, features]
    if includeZeroes == 'all':
        ds_ninnov = ds_ninnov[(ds_ninnov !=0).all(axis=1)]
        ds_innov = ds_innov[(ds_innov !=0).all(axis=1)]
    if includeZeroes == 'any':
        ds_ninnov = ds_ninnov[(ds_ninnov !=0).any(axis=1)]
        ds_innov = ds_innov[(ds_innov !=0).any(axis=1)]
    if (len(ds_ninnov) < 4 or len(ds_innov) < 4):
        print(f"Not enough nonzero samples exist for features {features}. Try setting 'includeZeroes' to None or any.")
        return

    xmin, ymin, zmin = [min(dataSet[f]) for f in features]
    xmax, ymax, zmax = [max(dataSet[f]) for f in features]
    
    # compute KDE's
    x_ninnov, y_ninnov, z_ninnov = [ds_ninnov[f] for f in features]
    f_ninnov = np.vstack([x_ninnov, y_ninnov, z_ninnov])
    x_innov, y_innov, z_innov = [ds_innov[f] for f in features]
    f_innov = np.vstack([x_innov, y_innov, z_innov])
    try:
        kde_ninnov = gaussian_kde(f_ninnov)
        kde_innov = gaussian_kde(f_innov)
    except np.linalg.LinAlgError:
        print(f"Not enough nonzero samples exist for features {features}. Try setting 'includeZeroes' to None or any.")
        return
    
    # show plot and save animation
    if showPlot:
        plt.figure(figsize=figSize)
        plt.title(f"3D space of features: {features}")
        ax = plt.axes(projection='3d')
        colormap = ['blue']*len(x_ninnov) + ['red']*len(x_innov)
        ax.scatter3D(pandas.concat([x_ninnov, x_innov]),
                    pandas.concat([y_ninnov, y_innov]),
                    pandas.concat([z_ninnov, z_innov]), c=colormap)
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_zlim([zmin, zmax])
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(features[2])
        blue_patch = mpatches.Patch(color='blue', label='Non-innovative')
        red_patch = mpatches.Patch(color='red', label='Innovative')
        plt.legend(handles=[blue_patch,red_patch], loc='upper right')

        if createAnimation:
            frames = []
            frames_dir = f"{path}/figures/webs/frames"
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir)
            for file in os.listdir(frames_dir):
                os.remove(f"{frames_dir}/{file}")
            for i in range(n_frames):
                ax.view_init(elev=20, azim=i*(360/n_frames))
                plt.savefig(f'{frames_dir}/frame_{i:03}.png', bbox_inches='tight')
            for file in os.listdir(frames_dir):
                frames.append(Image.open(f"{frames_dir}/{file}"))
            imageio.mimsave(f'{path}/figures/webs/3D_dist_{features}.gif', frames)
            for file in os.listdir(frames_dir):
                os.remove(f"{frames_dir}/{file}")
            os.rmdir(frames_dir)

    return kde_ninnov, kde_innov

def kde_n(dataSet, features, includeZeroes='none'):
    '''
    Uses Gaussian Kernel Density Estimation to compute two nD probability
    densities of n selected features from the dataset, one for both
    classes.
    'features' requires a list of features.
    'includeZeroes' eliminates zero values:
        - 'all' only considers companies with both features nonzero;
        - 'any' only considers companies with at least one feature nonzero;
        - 'none' does not eliminate zero values.
    '''

    ds_ninnov = dataSet.loc[dataSet['Innov'] == 0, features]
    ds_innov = dataSet.loc[dataSet['Innov'] == 1, features]
    if includeZeroes == 'all':
        ds_ninnov = ds_ninnov[(ds_ninnov !=0).all(axis=1)]
        ds_innov = ds_innov[(ds_innov !=0).all(axis=1)]
    if includeZeroes == 'any':
        ds_ninnov = ds_ninnov[(ds_ninnov !=0).any(axis=1)]
        ds_innov = ds_innov[(ds_innov !=0).any(axis=1)]

    x_ninnov = ds_ninnov[features]
    x_innov = ds_innov[features]

    f_ninnov = np.vstack([x_ninnov[f] for f in features])
    f_innov = np.vstack([x_innov[f] for f in features])
    kde_ninnov = gaussian_kde(f_ninnov)
    kde_innov = gaussian_kde(f_innov)

    return kde_ninnov, kde_innov

def compute_overlap(dataSet, features, includeZeroes='none', samples=20_000):
    '''
    Computes the n-dimensional volume of the overlap between
    two n-dimensional probability densities of n features.
    This is achieved via Monte Carlo integration.
    'samples' is the number of random samples taken for the
    Monte Carlo approach. This should scale exponentially with
    the number of considered features to preserve accuracy.
    'includeZeroes' eliminates zero values:
        - 'all' only considers companies with both features nonzero;
        - 'any' only considers companies with at least one feature nonzero;
        - 'none' does not eliminate zero values.
    '''

    print(f"Computing Monte-Carlo overlap for feature(s): {features}.")
    n = len(features)
    ds_ninnov = dataSet.loc[dataSet['Innov'] == 0, features]
    ds_innov = dataSet.loc[dataSet['Innov'] == 1, features]
    if includeZeroes == 'all':
        ds_ninnov = ds_ninnov[(ds_ninnov !=0).all(axis=1)]
        ds_innov = ds_innov[(ds_innov !=0).all(axis=1)]
    if includeZeroes == 'any':
        ds_ninnov = ds_ninnov[(ds_ninnov !=0).any(axis=1)]
        ds_innov = ds_innov[(ds_innov !=0).any(axis=1)]
    if (len(ds_ninnov) < n+1 or len(ds_innov) < n+1):
        print(f"Not enough nonzero samples exist for features {features}. Try setting 'includeZeroes' to None or any.")
        return None, None, None, features
    
    try:
        kernel_ninnov, kernel_innov = kde_n(dataSet, features, includeZeroes)
    except np.linalg.LinAlgError:
        print(f"Not enough nonzero samples exist for features {features}. Try setting 'includeZeroes' to None or any.")
        return None, None, None, features
    
    mn = [min(dataSet[f]) for f in features]
    mx = [max(dataSet[f]) for f in features]

    l = [np.linspace(min,max,100) for (min,max) in zip(mn,mx)]
    X = np.meshgrid(*l)

    positions = np.vstack([x.ravel() for x in X])

    zmx_ninnov = max(kernel_ninnov.pdf(positions))
    zmx_innov = max(kernel_innov.pdf(positions))
    zmx = max(zmx_ninnov, zmx_innov)
    below_density_ninnov = 0
    below_density_innov = 0
    below_densities = 0
    i = 0

    dist_uniform = [Distribution(uniform(min,max-min)) for (min,max) in zip(mn,mx)]
    dist_uniform_z = Distribution(uniform(0,zmx))

    uniforms = [d.rvs(samples) for d in dist_uniform]
    z_uniforms = dist_uniform_z.rvs(samples)

    while (i < samples):
        x = [u[i] for u in uniforms]
        z = z_uniforms[i]

        z_ninnov = kernel_ninnov.pdf(x)
        z_innov = kernel_innov.pdf(x)
        if z <= z_ninnov:
            below_density_ninnov += 1
        if z <= z_innov:
            below_density_innov += 1
        if z <= min(z_ninnov, z_innov):
            below_densities += 1
        i += 1

    area_ninnov = ( np.prod([max-min for (min,max) in zip(mn,mx)]) * zmx * below_density_ninnov) / samples
    area_innov = ( np.prod([max-min for (min,max) in zip(mn,mx)]) * zmx * below_density_innov) / samples
    area_both = ( np.prod([max-min for (min,max) in zip(mn,mx)]) * zmx * below_densities) / samples

    return area_ninnov, area_innov, area_both, features

def compute_misclassification(dataSet, features, classifiers, includeZeroes='none'):
    '''
    Computes the 1D probability of misclassification of a
    randomly selected websites from the dataset from a given feature.
    'features' requires a list of features that will all be considered
    separately.
    'classifiers' is a list of values computed by the method 'kde_1' which
    represent some good values for features to distinguish the two
    classes of websites. These values are used to compute the misclassification
    probabilities for every feature if such a classifier was found.
    'includeZeroes' eliminates zero values:
        - 'all' only considers companies with both features nonzero;
        - 'any' only considers companies with at least one feature nonzero;
        - 'none' does not eliminate zero values.
    '''

    misclassification = {}
    for feature in features:
        c = classifiers[feature]
        if c[0] is None:
            print(f"Misclassification could not be computed for feature: {feature}.")
            misclassification[feature] = None
            continue
        
        kernel_ninnov, kernel_innov = kde_n(dataSet, [feature])
        nr_websites = len(ds_webs)
        nr_noninnovatives = len(ds_ninnov)
        nr_innovatives = len(ds_innov)
        if c[1] == 0: # feature value under classifier is classified as non-innovative
            misclassification[feature] = kernel_innov.integrate_box_1d(-1, c[0])*nr_innovatives/nr_websites + \
                                         kernel_ninnov.integrate_box_1d(c[0], 2)*nr_noninnovatives/nr_websites
        if c[1] == 1: # feature value under classifier is classified as innovative
            misclassification[feature] = kernel_ninnov.integrate_box_1d(-1, c[0])*nr_noninnovatives/nr_websites + \
                                         kernel_innov.integrate_box_1d(c[0], 2)*nr_innovatives/nr_websites

    return misclassification


# Read websites dataset
print("Reading websites dataset.")
ds_webs, ds_ninnov, ds_innov, features = read_ds_webs(f'{path}/data')
if not os.path.exists(f'{path}/figures/webs'):
    os.makedirs(f'{path}/figures/webs')
if not os.path.exists(f'{path}/results/webs'):
    os.makedirs(f'{path}/results/webs')


# Logistic regression, compute weights of features.
# Generate plot of weights in 'figures' folder.
# Save weights locally in 'results' folder.
print("Computing logistic regression.")
alg, acc, main_features = logistic_regression(ds_webs, showPlot=True, figSize=(16,10))
features = [y for (x,y) in main_features]


# Compute 1D Guassian KDE's of (non-)innovative websites distributions
# for every main feature. Generate plots and save the standard
# classifiers where the distributions intersect (if they can be found).
# In the 'classifiers.txt' file, the first value specifies
# its value, the second specifies whether an innovative company
# corresponds to the left of the classifier (0 = non-innovative, 1 = innovative)
# For clearer plots, it's a good idea to not include zero values.
print("Computing 1D KDE's.")
kde_ninnov, kde_innov, classifiers = kde_1(ds_webs, features, showPlot=True, includeZeroes=False)


# Compute 2D Gaussian KDE's of (non-)innovative websites distributions
# for all combinations of a selection of features. Generate plots.
# For clearer plots, it's a good idea to not include zero values.
# 'nz' indicates whether zero values should be included or not:
#   - 'all' only considers companies with both features nonzero;
#   - 'any' only considers companies with at least one feature nonzero;
#   - 'none' does not eliminate zero values.
nz = 'all'
selected_features = random.sample(features, 7)
print("Computing 2D KDE's for all pairs of features in:")
print(selected_features)
combos = combinations(selected_features, 2)
if use_multiprocessing:
    Parallel(n_jobs = num_cores) (delayed(kde_2)(ds_webs, c, showPlot=True, includeZeroes=nz) for c in combos)
if not use_multiprocessing:
    for c in combos:
        kde_2(ds_webs, c, showPlot=True, includeZeroes=nz)


# Compute 3D Gaussian KDE's of (non-)innovative websites distributions
# for all combinations of a selection of features. Generate animations.
# For clearer plots, it's a good idea to not include zero values.
# 'nz' indicates whether zero values should be included or not:
#   - 'all' only considers companies with both features nonzero;
#   - 'any' only considers companies with at least one feature nonzero;
#   - 'none' does not eliminate zero values.
nz = 'any'
triples = [('f1753', 'f1776', 'f117'), ('f692', 'f38', 'f117')]
print("Computing 3D KDE's for all triples of features in:")
print(triples)
for c in triples:
    kde_3(ds_webs, c, showPlot=True, includeZeroes=nz, createAnimation=True, n_frames=60)


# Compute 1D overlap of (non-)innovative websites distributions
# for every feature. Save the overlap values locally
# to the 'results' folder.
# 'nz' indicates whether zero values should be included or not:
#   - 'all' only considers companies with both features nonzero;
#   - 'any' only considers companies with at least one feature nonzero;
#   - 'none' does not eliminate zero values.
nz = 'all'
print("Computing overlap values via Monte-Carlo.")
overlaps = {}
if use_multiprocessing:
    res = Parallel(n_jobs = num_cores) (delayed(compute_overlap)(ds_webs, [f], includeZeroes=nz, samples=20_000) for f in features)
    overlaps = {d[0]:c for (a,b,c,d) in res}
if not use_multiprocessing:
    for f in features:
        _,_,overlap,_ = compute_overlap(ds_webs, [f], includeZeroes=nz, samples=20_000)
        overlaps[f] = overlap
f = open(f'{path}/results/webs/overlap.txt', 'w')
f.write(str(overlaps))
f.close()


# Compute 1D misclassification probabilities for
# every feature. Save the misclassification values
# locally to the 'results' folder.
print("Computing misclassification probabilities.")
misclassification = compute_misclassification(ds_webs, features, classifiers)
f = open(f'{path}/results/webs/misclassification.txt', 'w')
f.write(str(misclassification))
f.close()


# Creates a histogram of overlap values and
# misclassification probabilities for every feature.
misclassification_ = {f : (0 if v is None else v) for (f,v) in misclassification.items()}
plt.figure(figsize=(16,10))
ax = plt.axes()
ax.set_axisbelow(True)
plt.xticks(rotation=90)
plt.grid()
plt.title("Overlap and misclassification values for all features")
plt.bar(features, overlaps.values())
plt.bar(features, misclassification_.values(), alpha=0.7)
plt.subplots_adjust(bottom=0.2)
blue_patch = mpatches.Patch(color='blue', label='Area of overlap')
orange_patch = mpatches.Patch(color='orange', label='Probability of misclassification')
plt.legend(handles=[blue_patch,orange_patch])
plt.savefig(f'{path}/figures/webs/overlap_misclassification_values_histogram.png', bbox_inches='tight')
plt.close()