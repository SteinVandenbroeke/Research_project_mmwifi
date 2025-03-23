import csv
import math

with open("output/output_5Ghz_12_03.csv", mode='r') as file:
    data_counter = {}
    csvreader = csv.reader(file)
    for row in csvreader:
        if row[1] not in data_counter:
            data_counter[row[1]] = {}

        if row[2] not in data_counter[row[1]]:
            data_counter[row[1]][row[2]] = 0
        data_counter[row[1]][row[2]] += 1

    print(len(data_counter.keys()))
    least_measurements = math.inf
    most_measurements = 0
    total_measurements = 0
    for name, data in data_counter.items():
        print(name)
        print(len(data.keys()), data)
        for position, item in data.items():
            if item < 200:
                print(f"Not enghough data for position {position}")
            if item < least_measurements:
                least_measurements = item
            if item > most_measurements:
                most_measurements = item
            total_measurements += item
    print(f"least_measurements: {least_measurements}")
    print(f"most_measurements: {most_measurements}")
    print(f"total_measurements: {total_measurements}")
    print(f"avarge_measurements: {total_measurements/(20*15)}")
    print(f"avarge_measurements: {(total_measurements / (20 * 15))/2}")
