import pandas as pd 
import numpy as np
from sklearn.preprocessing import StandardScaler
from os import getcwd, path, mkdir
from Analysis import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap

dataframe_full = pd.read_csv(getcwd()+"/data/integrated_data_combined.csv")
list_headers = list(dataframe_full)
array_coordinates = dataframe_full.values[:,0:2]
array_sentiments = dataframe_full.values[:,3:8]
array_weather0 = np.array(dataframe_full.values[:,8:13], dtype=float)
set_weather1 = set(dataframe_full.values[:,13])
array_weather1 = np.zeros((dataframe_full.values.shape[0], len(set_weather1)))
array_weather2 = np.array(dataframe_full.values[:,14:17], dtype=float)
array_weather3 = np.array(dataframe_full.values[:,18:19], dtype=float)
array_weather4 = np.array(dataframe_full.values[:,19:21], dtype=float)
array_day_night = np.zeros((dataframe_full.values.shape[0],2))
array_phase24h = np.empty((dataframe_full.values.shape[0],1))

list_headers_coordinates = list_headers[0:2]
list_headers_sentiments = list_headers[3:8]
list_headers_weather0 = list_headers[8:13]
list_headers_weather1 = [list_headers[13]]
list_headers_weather2 = list_headers[14:17]
list_headers_weather3 = list_headers[18:19]
list_headers_weather4 = list_headers[19:21]
list_headers_day_night = ["day(_)", "night(_)"]


'''
=======================================================
	Pre-Processing of Location Coordinates
=======================================================
'''

list_coordinates = [str(coord[0])+' '+str(coord[1]) for coord in array_coordinates]
set_coordinates = set(list_coordinates)
array_coordinates_unique = np.array([[float(unq_coord.split()[0]),float(unq_coord.split()[1])] for unq_coord in set_coordinates])

'''
==================================================
	Cluster Analysis of Locations
==================================================
'''

kMeans_elbow_method_plot(array_coordinates_unique, 20, "Coordinates_kMeans_ElbowMethod.svg")

def create_background_map():
	plt.figure()
	background_map = Basemap(projection='stere', lat_0=+38.60, lon_0=-97.72, llcrnrlat=+22.00, urcrnrlat=+48.96, llcrnrlon=-122.68, urcrnrlon=-60.28, rsphere=6371200., resolution='l', area_thresh=10000)
	background_map.drawcoastlines()
	background_map.drawstates()                  
	background_map.drawcountries()
	parallels = np.arange(0.,90,10.)
	background_map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
	meridians = np.arange(180.,360.,10.)
	background_map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
	
	return background_map

scaler_coordinates = StandardScaler()
kBest_coordinates = 3
list_colours = ['r', 'g', 'b']
array_coordinates_scaled_unique = scaler_coordinates.fit_transform(array_coordinates_unique)
kMeans_coordinates = kMeans_model(array_coordinates_scaled_unique, kBest_coordinates)
array_kMeans_coordinate_labels = kMeans_coordinates[1]
array_kMeans_coordinate_cen = kMeans_coordinates[2]

array_kMeans_coordinate_colours = np.array([list_colours[label] for label in array_kMeans_coordinate_labels])

array_latitudes = array_coordinates_unique[:, 0]
array_longitudes = array_coordinates_unique[:, 1]
bckgr_map = create_background_map()
x, y = bckgr_map(array_longitudes, array_latitudes)
bckgr_map.scatter(x, y, color=array_kMeans_coordinate_colours, s=200.0, marker='o', alpha=0.7)
plt.savefig("LocationClusters.svg")

'''
=============================================
	Weather Data Pre-Processing	
=============================================
'''
# One-Hot Vectors from Weather Descriptions

dict_weather1 = {}
i_weather1 = 0
for desc_weather1 in set_weather1:
	dict_weather1[desc_weather1] = i_weather1
	i_weather1 += 1

i_weather1 = 0
for weather1 in dataframe_full.values[:,13]:
	array_weather1[i_weather1, dict_weather1[weather1]] = 1
	i_weather1 += 1

# Day/Night One-Hot Vectors and Phase Angle

array_time = dataframe_full.values[:,2]
array_sunrise = dataframe_full.values[:,14]
array_sunset = dataframe_full.values[:,15]
for i_day_night in range(dataframe_full.values.shape[0]):
	hours_day = array_sunset[i_day_night] - array_sunrise[i_day_night]
	hours_night = 24.0 - hours_day
	if (array_time[i_day_night] == array_sunrise[i_day_night]):	# Sunrise
		array_phase24h[i_day_night,0] = 0
	elif (array_time[i_day_night] == array_sunset[i_day_night]):	# Sunset
		array_phase24h[i_day_night,0] = np.pi
	elif (array_sunrise[i_day_night] < array_time[i_day_night]) and (array_time[i_day_night] < array_sunset[i_day_night]):	# Day
		array_day_night[i_day_night,0] = 1.0
		array_phase24h[i_day_night,0] = np.pi * (array_time[i_day_night] - array_sunrise[i_day_night])/hours_day
	else:	# Night
		array_day_night[i_day_night,1] = 1.0
		if array_time[i_day_night] < array_sunset[i_day_night]: # Past midnight
			array_time[i_day_night] += 24.0
		array_phase24h[i_day_night,0] = np.pi * (-1 + (array_time[i_day_night] - array_sunset[i_day_night])/hours_night)
		if array_time[i_day_night] >= 24.0: # Past midnight
			array_time[i_day_night] -= 24.0

# Scaled Combination of Weather Data for Cluster Analysis

scaler_weather0 = StandardScaler()
array_weather0 = scaler_weather0.fit_transform(array_weather0)
scaler_weather2 = StandardScaler()
array_weather2 = scaler_weather2.fit_transform(array_weather2)
scaler_weather3 = StandardScaler()
array_weather3 = scaler_weather3.fit_transform(array_weather3)
#scaler_phase24h = StandardScaler()
#array_phase24h = scaler_phase24h.fit_transform(array_phase24h) 

array_weather = np.empty((dataframe_full.shape[0], array_weather0.shape[1] + array_weather1.shape[1] + array_weather2.shape[1] + array_weather3.shape[1] + 2))

array_weather[:,0:array_weather0.shape[1]] = array_weather0
array_weather[:,array_weather0.shape[1]:array_weather0.shape[1]+array_weather1.shape[1]] = array_weather1
array_weather[:,array_weather0.shape[1]+array_weather1.shape[1]:array_weather0.shape[1]+array_weather1.shape[1]+array_weather2.shape[1]] = array_weather2
array_weather[:,array_weather0.shape[1]+array_weather1.shape[1]+array_weather2.shape[1]:array_weather0.shape[1]+array_weather1.shape[1]+array_weather2.shape[1]+array_weather3.shape[1]] = array_weather3
array_weather[:,array_weather0.shape[1]+array_weather1.shape[1]+array_weather2.shape[1]+array_weather3.shape[1]:array_weather0.shape[1]+array_weather1.shape[1]+array_weather2.shape[1]+array_weather3.shape[1]+2] = array_day_night

'''
=================================================
	Cluster Analysis of Weather Data
=================================================
'''

kMeans_elbow_method_plot(array_weather, 20, "Weather_kMeans_ElbowMethod.svg")

kBest_weather = 4

kMeans_weather = kMeans_model(array_weather, kBest_weather)
array_weather_cen0 = scaler_weather0.inverse_transform(kMeans_weather[2][:,0:array_weather0.shape[1]])
array_weather_cen1 = kMeans_weather[2][:,array_weather0.shape[1]:array_weather0.shape[1]+array_weather1.shape[1]]
array_weather_cen2 = scaler_weather2.inverse_transform(kMeans_weather[2][:,array_weather0.shape[1]+array_weather1.shape[1]:array_weather0.shape[1]+array_weather1.shape[1]+array_weather2.shape[1]])
array_weather_cen3 = scaler_weather3.inverse_transform(kMeans_weather[2][:,array_weather0.shape[1]+array_weather1.shape[1]+array_weather2.shape[1]:array_weather0.shape[1]+array_weather1.shape[1]+array_weather2.shape[1]+array_weather3.shape[1]])

list_expl_weather0 = []
for cen in array_weather_cen0:
	weather0_expl = [ (list_headers_weather0[i_w0], cen[i_w0]) for i_w0 in range(len(list_headers_weather0)) ]
	print weather0_expl
	list_expl_weather0.append(weather0_expl)

list_expl_weather1 = []
for cen in array_weather_cen1:
	weather1_expl = sorted([ (desc, cen[dict_weather1[desc]]) for desc in dict_weather1.keys() ], key=lambda x: x[1], reverse=True)
	print weather1_expl
	list_expl_weather1.append(weather1_expl)

list_expl_weather2 = []
for cen in array_weather_cen2:
	weather2_expl = [ (list_headers_weather2[i_w2], cen[i_w2]) for i_w2 in range(len(list_headers_weather2)) ]
	print weather2_expl
	list_expl_weather2.append(weather2_expl)

list_expl_weather3 = []
for cen in array_weather_cen3:
	weather3_expl = [ (list_headers_weather3[i_w3], cen[i_w3]) for i_w3 in range(len(list_headers_weather3)) ]
	print weather3_expl
	list_expl_weather3.append(weather3_expl)


'''
================================================================
	Analysis: Tweet Sentiments vs. Circadian Rhythm
================================================================
'''

num_divs_phase24h = 100
delta_phase24h = 2*np.pi/num_divs_phase24h
array_index_phase24h = np.array(np.ceil((array_phase24h[:,0] + np.pi)/delta_phase24h)-1, dtype=int)

nested_list_AFINN = []
nested_list_comp = []
nested_list_neg = []
nested_list_neu = []
nested_list_pos = []

array_AFINN = array_sentiments[:,0]
array_comp = array_sentiments[:,1]
array_neg = array_sentiments[:,2]
array_neu = array_sentiments[:,3]
array_pos = array_sentiments[:,4]                                            

for i_phase24h in range(num_divs_phase24h):                                         
	nested_list_AFINN.append([])             
	nested_list_comp.append([])                              
	nested_list_neg.append([])                             
	nested_list_neu.append([])                           
	nested_list_pos.append([]) 

for i_data in range(dataframe_full.values.shape[0]):                                
	i_phase24h = array_index_phase24h[i_data]
	nested_list_AFINN[i_phase24h].append(array_AFINN[i_data])
	nested_list_comp[i_phase24h].append(array_comp[i_data])
	nested_list_neg[i_phase24h].append(array_neg[i_data])
	nested_list_neu[i_phase24h].append(array_neu[i_data])
	nested_list_pos[i_phase24h].append(array_pos[i_data])

array_sample_phase24h = np.arange(-np.pi+0.5*delta_phase24h, +np.pi, delta_phase24h)

list_avg_AFINN = []
list_avg_comp = []
list_avg_neg = []
list_avg_neu = []
list_avg_pos = []

list_stdev_AFINN = []
list_stdev_comp = []
list_stdev_neg = []
list_stdev_neu = []
list_stdev_pos = []

for i_phase24h in range(num_divs_phase24h):
	list_avg_AFINN.append(np.mean(nested_list_AFINN[i_phase24h]))
	list_avg_comp.append(np.mean(nested_list_comp[i_phase24h]))
	list_avg_neg.append(np.mean(nested_list_neg[i_phase24h]))
	list_avg_neu.append(np.mean(nested_list_neu[i_phase24h]))
	list_avg_pos.append(np.mean(nested_list_pos[i_phase24h]))
	list_stdev_AFINN.append(np.std(nested_list_AFINN[i_phase24h]))
	list_stdev_comp.append(np.std(nested_list_comp[i_phase24h]))
	list_stdev_neg.append(np.std(nested_list_neg[i_phase24h]))
	list_stdev_neu.append(np.std(nested_list_neu[i_phase24h]))
	list_stdev_pos.append(np.std(nested_list_pos[i_phase24h]))
	
phase24h_avg_fgr = plt.figure()
phase24h_avg_fgr.clf()
phase24h_avg_plt = phase24h_avg_fgr.add_subplot(1,1,1)
phase24h_avg_plt.plot(array_sample_phase24h, np.array(list_avg_AFINN), linestyle='-', marker='o', color='c', label="AFINN")
phase24h_avg_plt.plot(array_sample_phase24h, np.array(list_avg_comp), linestyle='-', marker='o', color='g', label="comp")
phase24h_avg_plt.plot(array_sample_phase24h, np.array(list_avg_neg), linestyle='-', marker='o', color='b', label="neg")
phase24h_avg_plt.plot(array_sample_phase24h, np.array(list_avg_neu), linestyle='-', marker='o', color='k', label="neu")
phase24h_avg_plt.plot(array_sample_phase24h, np.array(list_avg_pos), linestyle='-', marker='o', color='y', label="pos")
phase24h_avg_plt.set_xlabel("Journal Phase, $\delta$ [radians]")
phase24h_avg_plt.set_ylabel("Average Sentiment Score, $\overline{\Sigma}$ [dimensionless]")
phase24h_avg_plt.legend()
phase24h_avg_fgr.savefig("AverageSentiment_CircadianRhythm.svg")

phase24h_stdev_fgr = plt.figure()
phase24h_stdev_fgr.clf()
phase24h_stdev_plt = phase24h_stdev_fgr.add_subplot(1,1,1)
phase24h_stdev_plt.plot(array_sample_phase24h, np.array(list_stdev_AFINN), linestyle='-', marker='o', color='c', label="AFINN")
phase24h_stdev_plt.plot(array_sample_phase24h, np.array(list_stdev_comp), linestyle='-', marker='o', color='g', label="comp")
phase24h_stdev_plt.plot(array_sample_phase24h, np.array(list_stdev_neg), linestyle='-', marker='o', color='b', label="neg")
phase24h_stdev_plt.plot(array_sample_phase24h, np.array(list_stdev_neu), linestyle='-', marker='o', color='k', label="neu")
phase24h_stdev_plt.plot(array_sample_phase24h, np.array(list_stdev_pos), linestyle='-', marker='o', color='y', label="pos")
phase24h_stdev_plt.set_xlabel("Journal Phase, $\delta$ [radians]")
phase24h_stdev_plt.set_ylabel("Standard Deviation of Sentiment Score, $\sigma_{\Sigma}$ [dimensionless]")
phase24h_stdev_plt.legend()
phase24h_stdev_fgr.savefig("StandardDeviationSentiment_CircadianRhythm.svg")

def average_smoothing(input_list, window_size=3):
	output_list = []
	for i_data in range(len(input_list)):
		window_array = np.array(range(i_data-window_size, i_data+window_size+1), dtype=int) % len(input_list)
		sum_window = 0.0
		for i_window in window_array:
			sum_window += input_list[i_window]
		output_list.append( sum_window / (2*window_size + 1) )
	return np.array(output_list, dtype=float)

phase24h_avg_fgr = plt.figure()
phase24h_avg_fgr.clf()
phase24h_avg_plt = phase24h_avg_fgr.add_subplot(1,1,1)
phase24h_avg_plt.plot(array_sample_phase24h, average_smoothing(list_avg_AFINN), linestyle='-', marker='o', color='c', label="AFINN")
phase24h_avg_plt.plot(array_sample_phase24h, average_smoothing(list_avg_comp), linestyle='-', marker='o', color='g', label="comp")
phase24h_avg_plt.plot(array_sample_phase24h, average_smoothing(list_avg_neg), linestyle='-', marker='o', color='b', label="neg")
phase24h_avg_plt.plot(array_sample_phase24h, average_smoothing(list_avg_neu), linestyle='-', marker='o', color='k', label="neu")
phase24h_avg_plt.plot(array_sample_phase24h, average_smoothing(list_avg_pos), linestyle='-', marker='o', color='y', label="pos")
phase24h_avg_plt.set_xlabel("Journal Phase, $\delta$ [radians]")
phase24h_avg_plt.set_ylabel("Smoothed Average Sentiment Score, $\overline{\Sigma}$ [dimensionless]")
phase24h_avg_plt.legend()
phase24h_avg_fgr.savefig("AverageSentimentSmoothed_CircadianRhythm.svg")

phase24h_avg_fgr = plt.figure()
phase24h_avg_fgr.clf()
phase24h_avg_plt = phase24h_avg_fgr.add_subplot(1,1,1)
phase24h_avg_plt.plot(array_sample_phase24h, average_smoothing(list_avg_AFINN), linestyle='-', marker='o', color='c', label="AFINN")
phase24h_avg_plt.set_xlabel("Journal Phase, $\delta$ [radians]")
phase24h_avg_plt.set_ylabel("Smoothed Average Sentiment Score, $\overline{\Sigma}$ [dimensionless]")
phase24h_avg_plt.legend()
phase24h_avg_fgr.savefig("AverageSentimentSmoothed_CircadianRhythm_AFINN.svg")

phase24h_avg_fgr = plt.figure()
phase24h_avg_fgr.clf()
phase24h_avg_plt = phase24h_avg_fgr.add_subplot(1,1,1)
phase24h_avg_plt.plot(array_sample_phase24h, average_smoothing(list_avg_comp), linestyle='-', marker='o', color='g', label="comp")
phase24h_avg_plt.set_xlabel("Journal Phase, $\delta$ [radians]")
phase24h_avg_plt.set_ylabel("Smoothed Average Sentiment Score, $\overline{\Sigma}$ [dimensionless]")
phase24h_avg_plt.legend()
phase24h_avg_fgr.savefig("AverageSentimentSmoothed_CircadianRhythm_compNLTK.svg")

'''
====================================================================
	Analysis: Weather Data Principal Component Analysis
====================================================================
'''

pca_elbow_method_plot(array_weather, 19, "PCA_WeatherData.svg")
weather_components_PCA = pca_model(array_weather, 5)[1] 
list_headers_weather = list_headers_weather0 + list(dict_weather1.keys()) + list_headers_weather2 + list_headers_weather3 + list_headers_day_night

weather_components_explained = []
for i_component in range(weather_components_PCA.shape[0]):
	component_vector = weather_components_PCA[i_component]
	explained_component_raw = sorted([(list_headers_weather[i_basis], np.abs(component_vector[i_basis]), np.abs(component_vector[i_basis])/component_vector[i_basis]) for i_basis in range(weather_components_PCA.shape[1])], key=lambda x: x[1], reverse=True)
	explained_component = [(explained_component_raw[i_tuple][0], explained_component_raw[i_tuple][1]*explained_component_raw[i_tuple][2]) for i_tuple in range(len(explained_component_raw))]
	weather_components_explained.append(explained_component)
print weather_components_explained

'''
=========================================================================================================
	Scatter Plots of Negative-Sentiment Tweets -- Georgraphical Distributions by Weather Type
=========================================================================================================
'''

# Dictionary Structure with City Data: Key = City Name, Value = [State, (Latitude, Longitude), UTC_offset, OpenWeatherMap_city_code]
dict_city_coordinate =	{"new york": (40.6643, -73.9385),
			"los angeles": (34.0194, -118.4108),
			"chicago": (41.8376, -87.6818),
			"houston": (29.7805, -95.3863),
			"phoenix": (33.5722, -112.0880),
			"philadelphia": (40.0094, -75.1333),
			"san antonio": (29.4724, -98.5251),
			"san diego": (32.8153, -117.1350),
			"dallas": (32.7757, -96.7967),
			"san Jose": (37.2969, -121.8193),
			"austin": (30.3072, -97.7560),
			"jacksonville": (30.3370, -81.6613),
			"san francisco": (37.7751, -122.4193),
			"columbus": (39.9848, -82.9850),
			"indianapolis": (39.7767, -86.1459),
			"fort worth": (32.7795, -97.3463),
			"charlotte": (35.2087, -80.8307),
			"seattle": (47.6205, -122.3509),
			"denver": (39.7618, -104.8806),
			"el paso": (31.8484, -106.4270),
			"washington": (38.9041, -77.0171),
			"boston": (42.3320, -71.0202),
			"detroit": (42.3830, -83.1022),
			"nashville": (36.1718, -86.7850),
			"memphis": (35.1035, -89.9785),
			"portland, or": (45.5370, -122.6500),	# Special city key: to differentiate entry from Portland, ME
			"oklahoma city": (35.4671, -97.5137),
			"las vegas": (36.2277, -115.2640),
			"louisville": (38.1781, -85.6667),
			"baltimore": (39.3002, -76.6105)}

def find_city_key(coordinates, city_coordinate_dict):
	for key in city_coordinate_dict.keys():
		if (city_coordinate_dict[key][0] == coordinates[0]) and (city_coordinate_dict[key][1] == coordinates[1]):
			return key
	return None

# Dictionary of Sentiment Data by City
# * City name as key
# * For each weather type, counts of negative-sentiment tweets and all tweets to be tallied
# * Each city to contain region class
dict_city_wthr_sntmt =	{"new york": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"los angeles": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"chicago": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"houston": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"phoenix": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"philadelphia": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"san antonio": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"san diego": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"dallas": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"san Jose": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"austin": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"jacksonville": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"san francisco": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"columbus": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"indianapolis": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"fort worth": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"charlotte": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"seattle": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"denver": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"el paso": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"washington": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"boston": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"detroit": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"nashville": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"memphis": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"portland, or": [np.zeros((kBest_weather, 2), dtype=float), -1],	# Special city key: to differentiate entry from Portland, ME
			"oklahoma city": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"las vegas": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"louisville": [np.zeros((kBest_weather, 2), dtype=float), -1],
			"baltimore": [np.zeros((kBest_weather, 2), dtype=float), -1]}

#array_coordinates_scaled = scaler_coordinates.fit_transform(array_coordinates)
#kMeans_coordinate_all_labels = kMeans_model(array_coordinates_scaled, kBest_coordinates)[1]

# Updating Region Labels for City-Weather-Sentiment Dictionary
for i_coord in range(len(array_coordinates_unique)):
	key = find_city_key(array_coordinates_unique[i_coord], dict_city_coordinate)
	if key != None:
		dict_city_wthr_sntmt[key][1] = array_kMeans_coordinate_labels[i_coord]	
				
kMeans_weather_labels = kMeans_weather[1]
#print "# Diagnostic: kMeans_weather_labels = "
print kMeans_weather_labels

for i_data in range(dataframe_full.values.shape[0]):
	city_coordinates = array_coordinates[i_data]
	city_key = find_city_key(city_coordinates, dict_city_coordinate)
#	print "# Diagnostic: city_key = "+str(city_key)
	weather_label = kMeans_weather_labels[i_data]
#	print "# Diagnostic: weather_label = "+str(weather_label)
	if (city_key != None):
		dict_city_wthr_sntmt[city_key][0][weather_label][1] += 1	# Incrementing total tweet count for city + weather-type conbination
		if (array_AFINN[i_data] < 0) or (array_comp[i_data] < 0):
			dict_city_wthr_sntmt[city_key][0][weather_label][0] += 1	# Incrementing negative-sentimet tweet count for city + weather-type conbination

list_plot_markers = ['o', 'D', 's']

data_threshold = 10
list_neg_prop = []
for city_key in dict_city_wthr_sntmt.keys():
	for i_weather in range(kBest_weather):
		if (dict_city_wthr_sntmt[city_key][0][i_weather][1] > data_threshold):		
			list_neg_prop.append(dict_city_wthr_sntmt[city_key][0][i_weather][0] / dict_city_wthr_sntmt[city_key][0][i_weather][1])	
min_neg_prop = min(list_neg_prop)
max_neg_prop = max(list_neg_prop)

for i_weather in range(kBest_weather):
	bg_map = create_background_map()
	list_x_wthr = []
	list_y_wthr = []
	list_colour_param_wthr = []
	list_size_wthr = []
	for i_region in range(kBest_coordinates):
		for city_key in dict_city_wthr_sntmt.keys():
			if (dict_city_wthr_sntmt[city_key][1] == i_region) and (dict_city_wthr_sntmt[city_key][0][i_weather][1] > data_threshold):
				neg_tweet_proportion = dict_city_wthr_sntmt[city_key][0][i_weather][0] / dict_city_wthr_sntmt[city_key][0][i_weather][1]
				x, y = bg_map(dict_city_coordinate[city_key][1], dict_city_coordinate[city_key][0])
				list_x_wthr.append(x)
				list_y_wthr.append(y)
				index_viridis = 255 - int(np.round((255 * (neg_tweet_proportion - min_neg_prop) / (max_neg_prop - min_neg_prop))))
				list_colour_param_wthr.append(neg_tweet_proportion)
				list_size_wthr.append(200.0+dict_city_wthr_sntmt[city_key][0][i_weather][1])
	sct_plt0 = bg_map.scatter(list_x_wthr, list_y_wthr, c=list_colour_param_wthr, edgecolors='r', s=list_size_wthr, marker=list_plot_markers[i_region], vmin=min_n
eg_prop, vmax=max_neg_prop, cmap=cm.viridis_r, alpha=1.0)
	sct_plt1 = bg_map.scatter(list_x_wthr, list_y_wthr, color='r', s=20.0, marker='o', alpha=1.0)
	plt.colorbar(sct_plt0)
	plt.savefig("WeatherType"+str(i_weather)+"_Sentiments.svg")

'''

'''

def load_text_data_to_list(data_file_name):
	data_list = []
	if path.exists(getcwd()+'/'+data_file_name):
		data_file = open(getcwd()+'/'+data_file_name, 'r')
		for line in data_file:
			data_list.append(line)
		data_file.close()
	else:
		print "# Warning [load_text_data_to_list(...)]: File \'"+data_file_name+"\' not found in current directory ("+getcwd()+')'
	return data_list

list_neg = load_text_data_to_list("neg.txt")
list_pos = load_text_data_to_list("pos.txt")
list_neu = load_text_data_to_list("neu.txt")

def process_nmf(data_list, num_features=1000, num_topics=20, num_top_words=10):
	nmf_model_instance, nmf_feature_names = nmf_model(data_list, num_features, num_topics)
	print_top_words(nmf_model_instance, nmf_feature_names, num_top_words)

def process_lda(data_list, num_features=1000, num_topics=20, num_top_words=20):
	lda_model_instance, lda_feature_names = lda_model(data_list, num_features, num_topics)
	print_top_words(lda_model_instance, lda_feature_names, num_top_words)
	
print "# Negative Topics (NMF)..."
process_nmf(list_neg)
print "# Positive Topics (NMF)..."
process_nmf(list_pos)
print "# Neutral Topics (NMF)..."
process_nmf(list_neu)


'''

'''

from AncilliaeHTML import *

def process_nmf_html(html_file, h2_text, data_list, num_features=1000, num_topics=20, num_top_words=10):
	nmf_model_instance, nmf_feature_names = nmf_model(data_list, num_features, num_topics)
	add_html_h2(html_file, h2_text)
	top_words_html(nmf_model_instance, nmf_feature_names, num_top_words, html_file)	

def process_lda_html(html_file, h2_text, data_list, num_features=1000, num_topics=20, num_top_words=10):
	lda_model_instance, lda_feature_names = lda_model(data_list, num_features, num_topics)
	add_html_h2(html_file, h2_text)
	top_words_html(lda_model_instance, lda_feature_names, num_top_words, html_file)	

def get_text(data_file_name):
	if path.exists(getcwd()+'/'+data_file_name):
		return open(getcwd()+'/'+data_file_name).read()
	return None

def create_html_dir(dir_structure):
	if dir_structure[0] == '/':
		dir_structure = dir_structure[1:]
	if dir_structure[-1] == '/':
		dir_structure = dir_structure[:-1]
	dir_names = dir_structure.split('/')
	current_dir = getcwd()
	for dir_name in dir_names:
		current_dir += '/' + dir_name
		if not(path.exists(current_dir)):
			mkdir(current_dir)

flask_app_main_dir = "/FlaskApp/"
flask_app_pages_subdir = "pages/"
flask_app_images_subdir = "images/"
flask_app_pages_dir = flask_app_main_dir + flask_app_pages_subdir
flask_app_images_dir = flask_app_main_dir + flask_app_pages_subdir + flask_app_images_subdir

create_html_dir(flask_app_pages_dir)
create_html_dir(flask_app_images_dir)

index_html_file_name = "TopicIndex.html"
index_html_file = init_html_file(getcwd()+flask_app_main_dir+index_html_file_name, "Index Page -- Tweet Sentiment Topics by City")	

for city in dict_city_coordinate.keys():
	if city == "san Jose":
		continue
	name_prefix = city.replace(' ', '_').replace(',', '')
	
#	flask_app_pages_dir = flask_app_pages_dir+name_prefix
		
	
	list_neg = load_text_data_to_list(name_prefix+"_full__neg.txt")
	list_pos = load_text_data_to_list(name_prefix+"_full__pos.txt")
	list_neu = load_text_data_to_list(name_prefix+"_full__neu.txt")

	text_neg = get_text(name_prefix+"_full__neg.txt")
	text_pos = get_text(name_prefix+"_full__pos.txt")
	text_neu = get_text(name_prefix+"_full__neu.txt") 

	city_html_file = init_html_file(getcwd()+flask_app_pages_dir+'/'+name_prefix, city)

	create_wordcloud_image(text_neg, getcwd()+flask_app_images_dir+'/'+"WordCloud__neg__"+name_prefix+".png")
	add_html_h2(city_html_file, "Word Cloud -- Negative Tweets")
	add_html_image(flask_app_images_subdir+"WordCloud__neg__"+name_prefix, city_html_file)	#add_html_image(flask_app_images_subdir+"WordCloud__neg__"+name_prefix+".png", city_html_file)
	create_wordcloud_image(text_pos, getcwd()+flask_app_images_dir+'/'+"WordCloud__pos__"+name_prefix+".png")
	add_html_h2(city_html_file, "Word Cloud -- Positive Tweets")
	add_html_image(flask_app_images_subdir+"WordCloud__pos__"+name_prefix, city_html_file)	# add_html_image(flask_app_images_subdir+"WordCloud__pos__"+name_prefix+".png", city_html_file)
	create_wordcloud_image(text_neu, getcwd()+flask_app_images_dir+'/'+"WordCloud__neu__"+name_prefix+".png")
	add_html_h2(city_html_file, "Word Cloud -- Neutral Tweets")
	add_html_image(flask_app_images_subdir+"WordCloud__neu__"+name_prefix, city_html_file)	# add_html_image(flask_app_images_subdir+"WordCloud__neu__"+name_prefix+".png", city_html_file)

	process_nmf_html(city_html_file, "NMF Model -- Negative Tweets", list_neg)
	process_nmf_html(city_html_file, "NMF Model -- Positive Tweets", list_pos)
	process_nmf_html(city_html_file, "NMF Model -- Neutral Tweets", list_neu)

	process_lda_html(city_html_file, "LDA Model -- Negative Tweets", list_neg)
	process_lda_html(city_html_file, "LDA Model -- Positive Tweets", list_pos)
	process_lda_html(city_html_file, "LDA Model -- Neutral Tweets", list_neu)
	
	end_html_file(city_html_file)
	add_html_link(index_html_file, flask_app_pages_subdir+name_prefix, city)

end_html_file(index_html_file)

