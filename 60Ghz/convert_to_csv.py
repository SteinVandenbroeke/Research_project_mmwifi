import csv
import math
import os
from cmath import phase
from datetime import datetime, timedelta
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
    """
    Converts csi data file to csv based on timings_csv (adds the person)
    :param datafolder:
    :param timings_csv:
    :return:
    """
    print("start evalutating data")
    data_out13 = []
    with open(timings_csv, mode='r') as file:
            csvreader = csv.reader(file)
            mesurement = 0
            for row in csvreader:
                person_name = row[0]
                start_time = datetime.fromtimestamp(float(row[1]))
                end_time = datetime.fromtimestamp(float(row[2]))
                mmwave_files = sorted(f for f in os.listdir(f"./{datafolder}/{person_name}/") if f.endswith("out13"))
                for filename in mmwave_files:
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
                            data_out13.append([mesurement, row[0], row[3], start_time.isoformat(), end_time.isoformat(), (time_csi).isoformat(), 32] + mag.tolist())
                            mag, phase, time_csi, bad_idxs = get_next_line(mmwave_file)
                            #print(mag, phase, time_csi, bad_idxs)

                        mesurement += 1
    data_out17 = []
    with open(timings_csv, mode='r') as file:
        csvreader = csv.reader(file)
        mesurement = 0
        for row in csvreader:
            person_name = row[0]
            start_time = datetime.fromtimestamp(float(row[1]))
            end_time = datetime.fromtimestamp(float(row[2]))
            mmwave_files = sorted(f for f in os.listdir(f"./{datafolder}/{person_name}/") if f.endswith("out17"))
            for filename in mmwave_files:
                print(filename)
                if ".csv" in filename:
                    continue
                with open(os.path.join(f"./{datafolder}/{person_name}/", filename), "r") as mmwave_file:
                    mag, phase, time_csi, bad_idxs = get_next_line(mmwave_file)

                    # print("amplitudes", len(mag[0]))
                    # print("phases", len(phase[0]))

                    while time_csi is not None and start_time > time_csi:
                        mag, phase, time_csi, bad_idxs = get_next_line(mmwave_file)

                    while time_csi is not None and start_time < time_csi < end_time:
                        print(row[0], row[3], start_time.isoformat(), end_time.isoformat(), (time_csi).isoformat(),
                              "save")
                        data_out17.append([mesurement, row[0], row[3], start_time.isoformat(), end_time.isoformat(),
                                           (time_csi).isoformat(), 32] + mag.tolist())
                        mag, phase, time_csi, bad_idxs = get_next_line(mmwave_file)
                        # print(mag, phase, time_csi, bad_idxs)

                    mesurement += 1
    print("done")

    with open('output/output_60Ghz_out13.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data_out13)

    with open('output/output_60Ghz_out17.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data_out17)

#Way to slow
def combine_out13_and_out17(out13, out17):
    print("start combining out13 and out17")
    combined_data = []
    used = []
    skip_count = 0
    with open(out13, mode='r') as out13_file:
            out_13_csvreader = csv.reader(out13_file)
            for out13_row in out_13_csvreader:
                print(out13_row[1])
                out13_time = datetime.fromisoformat(out13_row[5])
                with open(out17, mode='r') as out17_file:
                    out_17_csvreader = csv.reader(out17_file)
                    smallest_delta_time = None
                    best_out17_row = None
                    best_out_time_17 = None
                    best_out_time_13 = None
                    for i, out17_row in enumerate(out_17_csvreader):
                        if out17_row[1] != out13_row[1] or out_17_csvreader.line_num in used:
                            continue
                        out17_time = datetime.fromisoformat(out17_row[5])
                        delta_time = abs(out13_time - out17_time)
                        #print(f"{out_17_csvreader.line_num} delta time {delta_time} {out13_time} {out17_time}")
                        if smallest_delta_time == None or smallest_delta_time > delta_time:
                            best_out_time_17 = out17_time
                            best_out_time_13 = out13_time
                            smallest_delta_time = delta_time
                            best_out17_row = out17_row
                        else:
                            used.append(out_17_csvreader.line_num)
                            #rint(f"match on {abs(best_out_time_17 - best_out_time_13)}")
                            assert not (best_out_time_17 is None or best_out_time_13 is None), "Did not found a match"
                            #assert abs((best_out_time_17 - best_out_time_13).seconds) < 1, "Time difference between out13 and out17 is to big!"
                            if abs(best_out_time_17 - best_out_time_13).seconds > 1:
                                print("skip row")
                                skip_count += 1
                            else:
                                assert not (best_out_time_17 is None or best_out_time_13 is None), "Did not found a match"
                                combined_data.append(out13_row + best_out17_row)
                            break

    with open('output/output_60Ghz_combined.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(combined_data)

    print(f"removed {skip_count} row because could not match the time")

def combine_out13_and_out17_one_pass(out13, out17):
    print("start combining out13 and out17")
    combined_data = []
    skip_count = 0
    start_time = datetime.now()

    #load complete 60Ghz measurements of device 17 in memory
    with open(out17, mode='r') as out17_file:
        out_17_csvreader = csv.reader(out17_file)
        out_17_csvreader_list = list(out_17_csvreader)
        total_rows = len(out_17_csvreader_list)

    #loop over all 60Ghz measurements of device 13
    with open(out13, mode='r') as out13_file:
            out_13_csvreader = csv.reader(out13_file)
            index_out17 = 0
            out17_time = None
            for out13_row in out_13_csvreader:
                current_time = datetime.now()
                print(out13_row[1], out_13_csvreader.line_num, abs(current_time - start_time))

                #take the exact measurement time of device 13
                out13_time = datetime.fromisoformat(out13_row[5])
                smallest_delta_time = None
                best_out17_row = None
                best_out_time_17 = None
                best_out_time_13 = None
                delta_time = None

                #As long as the measurement time of device 17 is larger than device 13 decrease the index_out17 so that we take earlier measurement than the measurement of device 13
                while out17_time == None or out17_time > out13_time and index_out17 != 0:
                    index_out17 -= 10
                    index_out17 = max(index_out17, 0)
                    out17_row = out_17_csvreader_list[index_out17]
                    out17_time = datetime.fromisoformat(out17_row[5])

                #loop over the measurements of device17 (we always know here that when we enter the measurement time of device 17 is earlier than device 13)
                while index_out17 < total_rows:
                    out17_row = out_17_csvreader_list[index_out17]
                    out17_time = datetime.fromisoformat(out17_row[5])
                    delta_time = abs(out13_time - out17_time)

                    #If the delta time between the measurement of device 13 and device 17 is smaller save it as the new delta time, otherwise this is the smallest possible delta time (because the files are sorted on time)
                    if smallest_delta_time == None or smallest_delta_time > delta_time:
                        best_out_time_17 = out17_time
                        best_out_time_13 = out13_time
                        smallest_delta_time = delta_time
                        best_out17_row = out17_row
                    else:
                        print(f"match on {abs(best_out_time_17 - best_out_time_13)}")
                        assert not (best_out_time_17 is None or best_out_time_13 is None), "Did not found a match"
                        #assert abs((best_out_time_17 - best_out_time_13).seconds) < 1, "Time difference between out13 and out17 is to big!"

                        #check whether the delta time is larger than 1s then there is somthing wrong like there is no corresponding measurement for device 17
                        if abs(best_out_time_17 - best_out_time_13).seconds > 1:
                            print("skip row", index_out17, best_out_time_17, best_out_time_13, abs(best_out_time_17 - best_out_time_13))
                            skip_count += 1
                        else:
                            assert not (best_out_time_17 is None or best_out_time_13 is None), "Did not found a match"
                            combined_data.append([smallest_delta_time] + out13_row + [index_out17] + best_out17_row) #combine the that into one row
                        break

                    #take next measure
                    index_out17 += 1

    with open('output/output_60Ghz_combined.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(combined_data)

    print(f"removed {skip_count} row because could not match the time")

def check_all_times(file):
    """
    Checks whether all items are correctly ordered in time
    :param file:
    :return:
    """
    with open(file, mode='r') as out_file:
        csvreader = csv.reader(out_file)
        prev_time = datetime.fromtimestamp(0)
        for row in csvreader:
            if prev_time > datetime.fromisoformat(row[5]):
                assert "FOUT"
            prev_time = datetime.fromisoformat(row[5])

def remove_dubbel_used_out(csv_combined):
    """
    Only keeps the smallest delta times whenever a measurement was used more than once
    :param csv_combined:
    :return:
    """
    indexes_to_remove = []
    with open(csv_combined, mode='r') as out_file:
        out_csvreader = list(csv.reader(out_file))
        useditems = {}
        for i, out_row in enumerate(out_csvreader):
            out17_index = out_row[9]
            if out17_index not in useditems:
                useditems[out17_index] = (datetime.strptime(out_row[0], "%H:%M:%S.%f").time(), i)
            elif useditems[out17_index][0] < datetime.strptime(out_row[0], "%H:%M:%S.%f").time():
                indexes_to_remove.append(i)
            elif useditems[out17_index][0] > datetime.strptime(out_row[0], "%H:%M:%S.%f").time():
                indexes_to_remove.append(useditems[out17_index][1])
                useditems[out17_index] = (datetime.strptime(out_row[0], "%H:%M:%S.%f").time(), i)

        for index in sorted(indexes_to_remove, reverse=True):
            del out_csvreader[index]

        print(f"There are {len(indexes_to_remove)} items removed")

        with open('output/output_60Ghz_combined_cleaned.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(out_csvreader)

def print_largest_delta(csv_file):
    """
    Prints some delta time metrics
    :param csv_file:
    :return:
    """
    with open(csv_file, mode='r') as out_file:
        out_csvreader = list(csv.reader(out_file))
        useditems = {}
        for i, out_row in enumerate(out_csvreader):
            out17_index = out_row[9]
            if out17_index not in useditems:
                useditems[out17_index] = (datetime.strptime(out_row[0], "%H:%M:%S.%f").time(), i)

    with open(csv_file, mode='r') as out_file:
            out_csvreader = csv.reader(out_file)
            largest_delta_time = None
            time_deltas = []
            for out_row in out_csvreader:
                h, m, s = map(float, out_row[0].replace(":", " ").split())
                time_deltas.append(timedelta(hours=h, minutes=m, seconds=s))

                if largest_delta_time == None or largest_delta_time < datetime.strptime(out_row[0], "%H:%M:%S.%f"):
                    largest_delta_time = datetime.strptime(out_row[0], "%H:%M:%S.%f")

            # Calculate total time
            total_time = sum(time_deltas, timedelta())

            # Calculate average time
            average_time = total_time / len(time_deltas)

            print(f"Largest delta time: {largest_delta_time.time()}")
            print(f"Total time error: {total_time}")
            print(f"Avg time error: {average_time}")




#sorry file paths are hard coded
csi_data_to_csv("unprocessed_data/locations", "../possition_timings.csv")
check_all_times('output/output_60Ghz_out13.csv')
check_all_times('output/output_60Ghz_out17.csv')
combine_out13_and_out17_one_pass('output/output_60Ghz_out13.csv', 'output/output_60Ghz_out17.csv')
remove_dubbel_used_out("output/output_60Ghz_combined.csv")
print_largest_delta("output/output_60Ghz_combined_cleaned.csv")