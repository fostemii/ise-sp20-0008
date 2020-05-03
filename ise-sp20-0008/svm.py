#This is the file that will download our dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from flask import send_file

url = 'https://docs.google.com/spreadsheets/d/1HDE_1BEJ9CNQY8y4oqxyGXWtiYc0XmHx2DHQbWf5H1c/export?format=csv&id=1HDE_1BEJ9CNQY8y4oqxyGXWtiYc0XmHx2DHQbWf5H1c&gid=0'

code_dir = os.path.dirname(__file__)

def get_data(filename):
	dir_loc = os.getcwd()
	filename = '/data/data.csv'
	#file = '/Users/michael/Downloads/e_data.csv'
	file = str(dir_loc+filename)
	energy_data = pd.read_csv(file)

	file1 = open(dir_loc+filename, 'r') 
	lines = file1.readlines() 

	year = []
	for i in energy_data["Year"]:
	    j = i[:-3]
	    k = int(j.replace("-", ''))
	    year.append(k)
    
	arr = []
	j = 0
	for i in year:
	    arr.append(j)
	    j = j + 1
	return(energy_data)

def training_data(col):
	scaler_data = StandardScaler()
	energy_data = get_data(code_dir + '/data/data.csv')
	#col = "FossilFuels"
	scale_data = scaler_data.fit_transform(energy_data[col].values.reshape(-1,1))
	x = StandardScaler().fit_transform(energy_data.drop(columns = ["Year", col, "Petroleum Coke", "All fuels (utility-scale)", "coal", 
                                       "petroleum liquids", "Petroleum Coke","Nature Gas", "Other Gases", 
                                       "wood and wood-derived fuels", "other biomass", "Other", "all solar", 
                                       "small-scale solar photovoltaic"]))
	x_train, x_test, y_train, y_test = train_test_split(x, scale_data, random_state = 1)
	return(x_train, x_test, y_train, y_test)

def gen_scatter(arg1, arg2, arg3):
    plot1 = svr_scatter(arg1, arg2, arg3)
    return send_file(plot1, attachment_filename = 'plot1.png', mimetype = 'image/png')

def gen_plot(arg1, arg2, arg3):
	plot2 = svr_plot(arg1, arg2, arg3)
	return send_file(plot2, attachment_filename = 'plot2.png', mimetype = 'image/png')

def svr_scatter(data_sel, k, c):
	x_train, x_test, y_train, y_test = training_data(data_sel)
	model = SVR(kernel = k, C = int(c), gamma = .01)
	model.fit(x_train, y_train)
	#time = arr[:58]
	prediction=model.predict(x_test)
	plt.scatter(prediction, y_test)
	plt.xlabel(data_sel)
	plt.ylabel("Other energy sources")
	plt.title("Standardized Energy Production")
	bytes_image = io.BytesIO()
	plt.savefig(bytes_image, format='png')
	bytes_image.seek(0)
	return bytes_image
	#return send_file(plot, attachment_filename = 'plot.png', mimetype = 'image/png')
	#path = os.path.dirname(__file__)
	#return(path)

def svr_plot(data_sel, k, c):
	x_train, x_test, y_train, y_test = training_data(data_sel)
	model = SVR(kernel = k, C = int(c), gamma = .01)
	model.fit(x_train, y_train)
	e = get_data('/data/data.csv')
	arr = time_arr(e)
	time = arr[:58]
	prediction=model.predict(x_test)
	plt.plot(time, prediction)
	plt.plot(time, y_test, 'r')
	bytes_image = io.BytesIO()
	plt.savefig(bytes_image, format='png')
	bytes_image.seek(0)
	return bytes_image


def time_arr(data):
	year = []
	for i in data["Year"]:
		j = i[:-3]
		k = int(j.replace("-", ''))
		year.append(k)

	arr = []
	j = 0
	for i in year:
		arr.append(j)
		j = j + 1
	return(arr)

