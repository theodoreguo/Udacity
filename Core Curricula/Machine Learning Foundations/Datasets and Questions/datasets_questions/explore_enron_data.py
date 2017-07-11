#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
records_num = len(enron_data)
print("Size of the Enron dataset: %i" %records_num)
print("Features In The Enron Dataset: %i" %len(enron_data["SKILLING JEFFREY K"]))

poi_count = 0
for key, value in enron_data.items():
    if value['poi'] == 1:
        poi_count += 1;
print("POIs in the Enron data: %i" %poi_count)

poi_names = open("../final_project/poi_names.txt")
poi_names = poi_names.readlines()
print("How many POIs exist: %i" %(len(poi_names)-2))

print("What is the total value of the stock belonging to James Prentice: %i" %enron_data["PRENTICE JAMES"]["total_stock_value"])

print("How many email messages do we have from Wesley Colwell to persons of interest: %i" %enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])

print("What's the value of stock options exercised by Jeffrey K Skilling: %i" %enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])

print("The total of the one who took the most money: %i" %max(enron_data["SKILLING JEFFREY K"]["total_payments"], enron_data["FASTOW ANDREW S"]["total_payments"], enron_data["LAY KENNETH L"]["total_payments"]))

salary_count = 0
for key, value in enron_data.items():
    if value["salary"] != "NaN":
        salary_count += 1;
print("The number of people whose salary is not null: %i" %salary_count)

email_count = 0
for key, value in enron_data.items():
    if value["email_address"] != "NaN":
        email_count += 1;
print("The number of people whose email is not null: %i" %email_count)

total_pay_count = 0
for key, value in enron_data.items():
    if value['total_payments'] == "NaN":
        total_pay_count += 1;
print("What percentage of people who have 'NaN' for their total payment: %.2f" % (total_pay_count*100/float(records_num)))

poi_payment_nan_count = 0
for key, value in enron_data.items():
    if value["poi"] == 1 and value["total_payments"]=="NaN":
        poi_payment_nan_count += 1;
print("What percentage of POIs who have 'NaN' for their total payment: %.2i" % (poi_payment_nan_count*100/float(poi_count)))




