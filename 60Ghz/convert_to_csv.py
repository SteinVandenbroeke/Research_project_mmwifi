import csv
import os
from cmath import phase
from datetime import datetime
from csi_parser import Parse_csi

def get_next_line(file):
    time_csi = []
    mag = None
    phase = None
    bad_idxs = None
    while type(time_csi) != datetime and len(time_csi) == 0:
        line = file.readline()
        if line == "":
            return None, None, None, None
        mag, phase, time_csi, bad_idxs = Parse_csi([line])
        if len(time_csi) == 0:
            continue
        time_csi = datetime.fromtimestamp(time_csi[0])
    return mag, phase, time_csi, bad_idxs

def csi_data_to_csv(datafolder, timings_csv):
    print("start evalutating data")
    data = []
    with open(timings_csv, mode='r') as file:
            csvreader = csv.reader(file)
            mesurement = 0
            for row in csvreader:
                person_name = row[0]
                start_time = datetime.fromtimestamp(float(row[1]))
                end_time = datetime.fromtimestamp(float(row[2]))
                for filename in sorted(os.listdir(f"./{datafolder}/{person_name}/")):
                    print(filename)
                    if ".csv" in filename:
                        continue
                    with open(os.path.join(f"./{datafolder}/{person_name}/", filename), "r") as mmwave_file:
                        mag, phase, time_csi, bad_idxs = get_next_line(mmwave_file)

                        #print("amplitudes", len(mag[0]))
                        #print("phases", len(phase[0]))

                        while time_csi is not None and start_time > time_csi:
                            mag, phase, time_csi, bad_idxs = get_next_line(mmwave_file)

                        while time_csi is not None and start_time < time_csi < end_time:
                            print(row[0], row[3], start_time.isoformat(), end_time.isoformat(), (time_csi).isoformat(), "save")
                            data.append([mesurement, row[0], row[3], start_time.isoformat(), end_time.isoformat(), (time_csi).isoformat(), 32] + mag.tolist())
                            mag, phase, time_csi, bad_idxs = get_next_line(mmwave_file)
                            #print(mag, phase, time_csi, bad_idxs)

                        mesurement += 1
    print("done")

    with open('output/output_5Ghz_12_03.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)

csi_data_to_csv("positions", "timing.csv")