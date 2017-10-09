# Necessary Package to JSON Data
try:
	import json
except ImportError:
	import simplejson as json

# Necessary Methods from "twitter" library
from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
from OpenWeatherMapData import weather_data_sample	# Not a standard Python module; wrappers around the OpenWeatherMap API

# NLTK Sentiment Analysis Tools
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize

from random import shuffle

from time import time

from os import path, getcwd, mkdir

# Variables Containing User Credentials to Access Twitter API 
ACCESS_TOKEN = '3713642416-yGze9AJCWjF0mRz3fMQ4TSGkhFgaWzBFFWEdbRu'
ACCESS_SECRET = 'f5etwHXCNisGbMLCGYJh89mCPtrDoFjUY2dWxDTLiciiu'
CONSUMER_KEY = 'xYkhuh9vwJaJoZIDEgNM7D8TR'
CONSUMER_SECRET = 'SgKRmrX0reYtOz7n7LSsjeFsx2xkCW1CLDbqBBavow1NMHdjBq'

oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)

# Connection to Twitter Streaming API
twitter_stream = TwitterStream(auth=oauth)

# Dictionary Structure with City Data: Key = City Name, Value = [State, (Latitude, Longitude), UTC_offset, OpenWeatherMap_city_code]
city_data =	{"new york": ["ny", (40.6643, -73.9385), -4, 5128581],
		"los angeles": ["ca", (34.0194, -118.4108), -7, 5368361],
		"chicago": ["il", (41.8376, -87.6818), -5, 4887398],
		"houston": ["tx", (29.7805, -95.3863), -5, 4699066],
		"phoenix": ["az", (33.5722, -112.0880), -7, 5308655],
		"philadelphia": ["pa", (40.0094, -75.1333), -4, 4560349],
		"san antonio": ["tx", (29.4724, -98.5251), -5, 4726206],
		"san diego": ["ca", (32.8153, -117.1350), -7, 5391811],
		"dallas": ["tx", (32.7757, -96.7967), -5, 4684888],
		"san Jose": ["ca", (37.2969, -121.8193), -7, 5392171],
		"austin": ["tx", (30.3072, -97.7560), -5, 4671654],
		"jacksonville": ["fl", (30.3370, -81.6613), -5, 4160021],
		"san francisco": ["ca", (37.7751, -122.4193), -7, 5391959],
		"columbus": ["oh", (39.9848, -82.9850), -4, 4509177],
		"indianapolis": ["in", (39.7767, -86.1459), -4, 4259418],
		"fort worth": ["tx", (32.7795, -97.3463), -5, 4691930],
		"charlotte": ["nc", (35.2087, -80.8307), -4, 4460243],
		"seattle": ["wa", (47.6205, -122.3509), -7, 5809844],
		"denver": ["co", (39.7618, -104.8806), -6, 5419384],
		"el paso": ["tx", (31.8484, -106.4270), -6, 5520993],
		"washington": ["dc", (38.9041, -77.0171), -4, 4140963],
		"boston": ["ma", (42.3320, -71.0202), -4, 4930956],
		"detroit": ["mi", (42.3830, -83.1022), -4, 4990729],
		"nashville": ["tn", (36.1718, -86.7850), -5, 4644585],
		"memphis": ["tn", (35.1035, -89.9785), -5, 4641239],
		"portland, or": ["or", (45.5370, -122.6500), -7, 5746545],	# Special city key: to differentiate entry from Portland, ME
		"oklahoma city": ["ok", (35.4671, -97.5137), -5, 4544349],
		"las vegas": ["nv", (36.2277, -115.2640), -7, 5506956],
		"louisville": ["ky", (38.1781, -85.6667), -4, 4299276],
		"baltimore": ["md", (39.3002, -76.6105), -4, 4347778]}

def create_data_files(city_names_list):
	pos_repository = {}	
	neu_repository = {}
	neg_repository = {}

	if not(path.exists(getcwd()+"/data")):
		mkdir(getcwd()+"/data")

	for city in city_names_list:
		name_prefix = city.replace(' ', '_').replace(',', '')
		
		pos_tweet_file = open(getcwd()+"/data/"+name_prefix+"__pos.txt",'w')
		neu_tweet_file = open(getcwd()+"/data/"+name_prefix+"__neu.txt",'w')
		neg_tweet_file = open(getcwd()+"/data/"+name_prefix+"__neg.txt",'w')

		pos_repository[city] = pos_tweet_file 	
		neu_repository[city] = neu_tweet_file
		neg_repository[city] = neg_tweet_file
	return pos_repository, neu_repository, neg_repository	

def close_data_files(file_dict):
	for key in file_dict.keys():
		file_dict[key].close()


def city_lookup(location):
	if (location == None):
		return None	
	
	entry_found = False
	for city in city_data.keys():
		state = city_data[city][0]
		entry_found = (city in location.lower()) and ((state in location.lower()) or ("usa" in location.lower())) and (len(location) - len(city) <= 8)
		if entry_found:
			return city
	return None

def term_minus_punctuation(term):
	punctuation_list=['.',',',';',':','\'','\"','(',')','!','`','?']	
	while((term!='') and (term[0] in punctuation_list)):
		term=term[1:]
	while((term!='') and (term[len(term)-1] in punctuation_list)):
		term=term[:-1]
	return term

def load_sentiment_dictionary(file_name):
	sentiment_file=open(file_name)
	dictionary={}
	for line in sentiment_file:
		term, score  = line.split("\t")
		term=unicode(term,"utf-8")
		dictionary[term]=int(score)
	sentiment_file.close()
	return dictionary

def sentiment_score(text,sentiment_dictionary):
	word_list=text.split(" ")
	score=0
	for word in word_list:
		if(term_minus_punctuation(word).lower() in sentiment_dictionary.keys()):
			score+=sentiment_dictionary[term_minus_punctuation(word).lower()]
	return score

def time_of_day(tweet_timestamp,GMT_offset):
	time_data = tweet_timestamp.split(" ")[3]
	hour = (int(time_data.split(':')[0])+GMT_offset)%24
	minute = int(time_data.split(':')[1])
	second = int(time_data.split(':')[2])
	return (hour + minute/60.0 + second/3600.0)


# Collecting Batches of Tweets Over a Specified Number of Hours
tweet_count_per_batch = 2000
data_hours = 24.0
connection_on = True

sentiment_data_file = "AFINN-111.txt"

if __name__=="__main__":

	repository_pos, repository_neu, repository_neg = create_data_files(city_data.keys())

	integrated_data_file = open(getcwd()+"/data/"+"integrated_data_combined.csv",'w')
	integrated_data_file.write("lat(degree),lon(degree),time(h),sntmt_AFINN(_),sntmt_NLTK_comp(_),sntmt_NLTK_neg(_),sntmt_NLTK_neu(_),sntmt_NLTK_pos(_),temp(C),temp_max(C),temp_min(C),humidity(%),pressure(hPa),sky(__),sunrise(h),sunset(h),wind(m/s),wind_deg(degree),cloudiness(%),rain_3h(mm),snow_3h(mm)\n")

	reference_data = load_sentiment_dictionary(sentiment_data_file)

	SI_analyser = SentimentIntensityAnalyzer()	

	start_time = time() 
	while(connection_on):
		# Sample of Public Data through Twitter
		iterator = twitter_stream.statuses.sample(language="en")
		
		if(connection_on == False):
			break	
		
		tweet_count = 0
		for tweet in iterator:
			city_key = city_lookup(tweet['user']['location'])

			if ('text' in tweet) and (tweet['user']['location'] != None) and (city_key != None):

				pos_tweet_file = repository_pos[city_key]
				neg_tweet_file = repository_neg[city_key]
				neu_tweet_file = repository_neu[city_key]

				print tweet['user']['name']
				print tweet['user']['location']
				#print tweet['coordinates']['coordinates']
				print tweet['created_at']
				print tweet['text']
				# print '\n'
			
				local_time = time_of_day(tweet['created_at'],city_data[city_key][2])
				print "Time value = "+str(local_time)

				tweet_score = 0
				tweet_text = tweet['text']
				if(tweet_text!=""):
					tweet_score=sentiment_score(tweet_text,reference_data)
				print "Sentiment score (AFINN) = "+str(tweet_score)

				tweet_sentences = tokenize.sent_tokenize(tweet_text)
				sntmt_sum = {"count":0, "compound":0.0, "neg":0.0, "neu":0.0, "pos":0.0}
				for sentence in tweet_sentences:
					sntmt_score = SI_analyser.polarity_scores(sentence)
					sntmt_sum["count"] += 1				
					for sntmt in sorted(sntmt_score):
						sntmt_sum[sntmt] += sntmt_score[sntmt]
				print "Sentiment score (NLTK) = ",
				for sntmt in sorted(sntmt_score):
					print str(sntmt)+": "+str(sntmt_sum[sntmt]/sntmt_sum["count"])+',',
				print "\n" 
			
				if (tweet_score > 0) or (sntmt_sum["compound"] > 0):
					pos_tweet_file.write(tweet['text'].encode("UTF-8"))
					pos_tweet_file.write('\n')				
				elif (tweet_score < 0) or (sntmt_sum["compound"] < 0):
					neg_tweet_file.write(tweet['text'].encode("UTF-8"))
					neg_tweet_file.write('\n')
				elif (tweet_score == 0) or (sntmt_sum["neu"] == 1.0):
					neu_tweet_file.write(tweet['text'].encode("UTF-8"))
					neu_tweet_file.write('\n')

				OWM_city_id = city_data[city_key][3]
				coordinates = city_data[city_key][1]	
	
				current_weather = weather_data_sample(OWM_city_id)
				integrated_data_file.write(str(coordinates[0])+','+str(coordinates[1])+','+str(local_time)+','+str(tweet_score)+','+str(sntmt_sum["compound"]/sntmt_sum["count"])+','+str(sntmt_sum["neg"]/sntmt_sum["count"])+','+str(sntmt_sum["neu"]/sntmt_sum["count"])+','+str(sntmt_sum["pos"]/sntmt_sum["count"])+','+str(current_weather["temp"])+','+str(current_weather["temp_max"])+','+str(current_weather["temp_min"])+','+str(current_weather["humidity"])+','+str(current_weather["pressure"])+','+str(current_weather["sky"])+','+str(current_weather["sunrise"])+','+str(current_weather["sunset"])+','+str(current_weather["wind"])+','+str(current_weather["wind_deg"])+','+str(current_weather["cloudiness"])+','+str(current_weather["rain_3h"])+','+str(current_weather["snow_3h"])+'\n')
		
				tweet_count += 1
				if (tweet_count >= tweet_count_per_batch):
					current_time = time()
					if ( (current_time - start_time)/3600.0 >= data_hours):
						connection_on = False
					break 	

	close_data_files(repository_pos)
	close_data_files(repository_neu)
	close_data_files(repository_neg)
	integrated_data_file.close()

