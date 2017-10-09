# Importing the necessary package to process data in JSON format
try:
	import json
except ImportError:
	import simplejson as json

try:
	from urllib.request import urlopen
except ImportError:
	from urllib import urlopen

import datetime

def time_converter(time):
    converted_time = datetime.datetime.fromtimestamp(
        int(time)
    ).strftime('%I:%M %p')
    return converted_time

def time_float(time):
    converted_time_12h = datetime.datetime.fromtimestamp(
        int(time)
    ).strftime('%I:%M %p')
    hour = int(converted_time_12h[0:2])
    if(converted_time_12h[-2:]=="PM"):
	hour += 12
	hour %= 24 
    return hour+int(converted_time_12h[3:5])/60.0


def url_builder(city_id):
    user_api = '329ab0ec0f5c1e89a4f6712e934bf7c9'  # Obtain yours form: http://openweathermap.org/
    unit = 'metric'  # For Fahrenheit use imperial, for Celsius use metric, and the default is Kelvin.
    api = 'http://api.openweathermap.org/data/2.5/weather?id='     # Search for your city ID here: http://bulk.openweathermap.org/sample/city.list.json.gz

    full_api_url = api + str(city_id) + '&mode=json&units=' + unit + '&APPID=' + user_api
    return full_api_url


def data_fetch(full_api_url):
    url = urlopen(full_api_url)
    output = url.read().decode('utf-8')
    raw_api_dict = json.loads(output)
    url.close()
    return raw_api_dict

def get_rain_data(raw_api_dict):
	if raw_api_dict.get('rain')==None:
		return float("NaN")
	return raw_api_dict.get('rain').get('3h')

def get_snow_data(raw_api_dict):
	if raw_api_dict.get('snow')==None:
		return float("NaN")
	return raw_api_dict.get('snow').get('3h')

def weather_data_ensemble(raw_api_dict):
    data = dict(
        city=raw_api_dict.get('name'),
        country=raw_api_dict.get('sys').get('country'),
        temp=raw_api_dict.get('main').get('temp'),
        temp_max=raw_api_dict.get('main').get('temp_max'),
        temp_min=raw_api_dict.get('main').get('temp_min'),
        humidity=raw_api_dict.get('main').get('humidity'),
        pressure=raw_api_dict.get('main').get('pressure'),
        sky=raw_api_dict['weather'][0]['main'],
        sunrise=time_float(raw_api_dict.get('sys').get('sunrise')),
        sunset=time_float(raw_api_dict.get('sys').get('sunset')),
        wind=raw_api_dict.get('wind').get('speed'),
        wind_deg=raw_api_dict.get('wind').get('deg'),
        dt=time_float(raw_api_dict.get('dt')),
        cloudiness=raw_api_dict.get('clouds').get('all'),
	rain_3h=get_rain_data(raw_api_dict),
	snow_3h=get_snow_data(raw_api_dict)
    )
    return data

def weather_data_sample(city_id):
	try:
		weather_data = data_fetch(url_builder(city_id))
#		print json.dumps(weather_data, indent=4)
		return weather_data_ensemble(weather_data)	
	except IOError:
		return dict(
        		city=None,
		        country=None,
		        temp=float("NaN"),
		        temp_max=float("NaN"),
		        temp_min=float("NaN"),
		        humidity=float("NaN"),
		        pressure=float("NaN"),
		        sky="",
		        sunrise=float("NaN"),
		        sunset=float("NaN"),
		        wind=float("NaN"),
		        wind_deg=float("NaN"),
		        dt=float("NaN"),
		        cloudiness=float("NaN"),
			rain_3h=float("NaN"),
			snow_3h=float("NaN")
    		)

def data_output(data):
    m_symbol = '\xb0' + 'C'
    print('---------------------------------------')
    print('Current weather in: {}, {}:'.format(data['city'], data['country']))
    print(data['temp'], m_symbol, data['sky'])
    print('Max: {}, Min: {}'.format(data['temp_max'], data['temp_min']))
    print('')
    print('Wind Speed: {} m/s, Direction: {} degrees'.format(data['wind'], data['wind_deg']))
    print('Humidity: {}%'.format(data['humidity']))
    print('Cloud: {}%'.format(data['cloudiness']))
    print('Pressure: {} hPa'.format(data['pressure']))
    print('Rain: {} mm'.format(data['rain_3h']))
    print('Snow: {} mm'.format(data['snow_3h']))
    print('Sunrise at: {}'.format(data['sunrise']))
    print('Sunset at: {}'.format(data['sunset']))
    print('')
    print('Last update from the server: {}'.format(data['dt']))
    print('---------------------------------------')

if __name__ == '__main__':
    try:
	OWM_city_id = 4930956	#2643743
	weather_data = data_fetch(url_builder(OWM_city_id))
        data_output(weather_data_ensemble(weather_data))
	print json.dumps(weather_data, indent=4)
    except IOError:
        print('no internet')
