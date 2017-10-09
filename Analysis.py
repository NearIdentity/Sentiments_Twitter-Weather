from sklearn.cluster import KMeans

def kMeans_model(X, k):
	model = KMeans(n_clusters=k)
	model.fit(X)
	y = model.labels_
	centres = model.cluster_centers_
	
	return model, y, centres


import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist
import seaborn
import numpy as np
from scipy.spatial.distance import cdist, pdist

def kMeans_elbow_method_plot(X_data, n, plot_name):
    kMeansVar = [KMeans(n_clusters=k).fit(X_data) for k in range(1, n)]
    centroids = [model.cluster_centers_ for model in kMeansVar]
    k_euclid = [cdist(X_data, cent) for cent in centroids]
    dist = [np.min(ke, axis=1) for ke in k_euclid]
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(X_data)**2)/X_data.shape[0]
    bss = tss - wcss
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    #ax.plot(np.arange(1,n), bss, marker='x', linestyle='-', color='g')
    plt.figure()
    plt.plot(bss)
    plt.xlabel("Number of Classes, k [dimensionless]")
    plt.ylabel("Distortion [arbitrary units]")
    plt.savefig(plot_name)


from sklearn.decomposition import PCA

def pca_elbow_method_plot(X, n, plot_name):
	num_components = range(1,n+1)
	pca_models = [PCA(n_components=l).fit(X) for l in num_components]
	expl_var_sums = [sum(model.explained_variance_ratio_) for model in pca_models]
	plt.figure()
	plt.xlabel("Number of Components, n [dimensionless]")
	plt.ylabel("Explained Variance Sum [arbitrary units]")
	plt.plot(num_components, expl_var_sums)
	plt.savefig(plot_name)
	
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

def pca_model(X, n):
	model = PCA(n_components=n).fit(X)
	return model, model.components_	

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

def lda_model(dataset, n_features, n_topics):
	tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                stop_words='english')
	tf = tf_vectorizer.fit_transform(dataset)
	lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
	lda.fit(tf)
	return lda, tf_vectorizer.get_feature_names()

def nmf_model(dataset, n_features, n_topics):
	tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
	tfidf = tfidf_vectorizer.fit_transform(dataset)
	nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5).fit(tfidf)
	tfidf_feature_names = tfidf_vectorizer.get_feature_names()
	return nmf, tfidf_feature_names	

from numpy import pi

def phase_24h(time_24h, time_sunrise, time_sunset):
	hours_day = time_sunset - time_sunrise
	hours_night = 24.0 - hours_day
	
	print "# Time = "+str(time_24h)
	print "# Sunrise = "+str(time_sunrise)	
	print "# Sunset = "+str(time_sunset)
	if time_24h == time_sunrise:	# Sunrise
		return 0
	elif time_24h == time_sunset:	# Sunset
		return pi
	elif time_24h > time_sunrise and time_24h < time_sunset:	# Day
		return +pi * (time_24h - time_sunrise)/hours_day
	else:	# Night
		if time_24h < time_sunset: # Past midnight
			time_24h += 24.0
		return +pi * (time_24h - time_sunrise)/hours_night

