import concurrent.futures

import csv
import os
import re
import subprocess
from datetime import datetime, timezone, timedelta
from functools import partial
from warnings import catch_warnings

from CSIKit.reader import get_reader
from CSIKit.util import csitools
#from CSIKit.filters.passband import lowpass
#from CSIKit.filters.statistical import running_mean
from CSIKit.util.filters import hampel
from CSIKit.tools.batch_graph import BatchGraph

import matplotlib.pyplot as plt
import numpy as np
import struct

from fontTools.misc.plistlib import end_data


def remove_last_packet(filename, new_filename):
    """
    Removes the last packet from a pcap file and saves the modified pcap as a new file.
    This function is handy when the last packet is incomplete or corrupted.
    :param filename: filename of the pcap file to modify
    :param new_filename: filename of the new (modified) pcap file
    :return:
    """
    from scapy.all import rdpcap, wrpcap

    packets = rdpcap(filename)

    if len(packets) > 0:
        packets = packets[:-1] # remove the last packet

    # Save the modified pcap to the new file
    wrpcap(new_filename, packets)

    print(f"Last packet removed. New file saved as {new_filename}")

def get_csi_data(pcap_file):
    reader = get_reader(pcap_file)
    csi_data: np.ndarray = None
    try:  # Exception handling for corrupted pcap files (e.g., incomplete packets)
        csi_data = reader.read_file(pcap_file, scaled=True)  # 'scaled=True' converts values to dB
    except (struct.error, ValueError) as e:
        print(f"Error reading file: {e}")
        try:
            new_filename = pcap_file.replace(".pcap", "_new.pcap")# Save the modified pcap as a new file
            if not os.path.exists(new_filename):
                print("Removing last packet and retrying...")
                remove_last_packet(pcap_file, new_filename)
            csi_data = reader.read_file(new_filename, scaled=True)  # 'scaled=True' converts values to dB
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    return csi_data

def extract_csi(pcap_file, metric="amplitude"):
    reader = get_reader(pcap_file)
    csi_data: np.ndarray = None
    try: # Exception handling for corrupted pcap files (e.g., incomplete packets)
        csi_data = reader.read_file(pcap_file, scaled=True) # 'scaled=True' converts values to dB
    except (struct.error, ValueError) as e:
        print(f"Error reading file: {e}")
        try:
            print("Removing last packet and retrying...")
            new_filename = pcap_file.replace(".pcap", "_new.pcap") # Save the modified pcap as a new file
            remove_last_packet(pcap_file, new_filename)
            csi_data = reader.read_file(new_filename, scaled=True)  # 'scaled=True' converts values to dB
        except Exception as e:
            print(f"Error reading file: {e}")
            print("If the error concerns an (integer) overflow error and you are running this on a Windows machine, "
                  "try running the script on a Unix-based OS.")
            return None
    print(csi_data.timestamps)
    csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric=metric)

    print(f"Extracted CSI Matrix Shape: {csi_matrix.shape}") # (num_frames, num_subcarriers, num_rx, num_tx)

    if csi_matrix.shape[2] == 1 and csi_matrix.shape[3] == 1:
        csi_matrix = np.squeeze(csi_matrix) # -> (num_frames, num_subcarriers)

    return csi_matrix

def extract_times(filename) -> (datetime, datetime):
    pattern = r"SYS-(\d{8}_\d{6}\.\d+)_ROUTER-(\d{8}_\d{6}\.\d+)"
    match = re.search(pattern, filename)

    if match:
        sys_time_str = match.group(1)
        ssh_time_str = match.group(2)

        print(ssh_time_str, sys_time_str)
        sys_time = datetime.strptime(sys_time_str[:22], "%Y%m%d_%H%M%S.%f")
        ssh_time = datetime.strptime(ssh_time_str[:22], "%Y%m%d_%H%M%S.%f")

        print(sys_time, ssh_time)

        return sys_time, ssh_time
    else:
        return None, None

def csi_data_to_csv(datafolder, timings_csv, run_for_person_name = None):
    print("start evalutating data")
    data = []
    with open(timings_csv, mode='r') as file:
        csvreader = list(csv.reader(file))
        file.close()
        mesurement = 0
        saved_files = {}
        person = None
        for row in csvreader:
            person_name = row[0].lower()
            print(person_name)
            if person != person_name:
                person = person_name
                saved_files = {}

            if person_name != run_for_person_name and run_for_person_name is not None:
                continue

            start_time = datetime.fromtimestamp(float(row[1]))
            end_time = datetime.fromtimestamp(float(row[2]))
            for filename in sorted(os.listdir(f"./{datafolder}/{person_name}/")):
                print(filename)
                pcap_file = os.path.join(f"./{datafolder}/{person_name}/", filename)
                system_time, ssh_time = extract_times(filename)

                if ".csv" in filename:
                    continue

                if pcap_file not in saved_files.keys():
                    csi_data = get_csi_data(pcap_file)
                    csi_i = 0
                    csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="amplitude")
                    saved_files[pcap_file] = csi_matrix, no_frames, no_subcarriers, csi_data
                else:
                    csi_i = 0
                    csi_matrix, no_frames, no_subcarriers, csi_data = saved_files[pcap_file]

                if csi_matrix.shape[2] == 1 and csi_matrix.shape[3] == 1:
                    csi_matrix = np.squeeze(csi_matrix)  # -> (num_frames, num_subcarriers)

                time_zone_diff = divmod((datetime.fromtimestamp(csi_data.timestamps[0]) - ssh_time).seconds, 3600)[0]
                delta_time = system_time - ssh_time - timedelta(hours=time_zone_diff)  # ajust for time zone difference
                print("delta time", delta_time)

                while csi_i < len(csi_data.timestamps) and start_time > datetime.fromtimestamp(csi_data.timestamps[csi_i]) + delta_time:
                    csi_i += 1

                while csi_i < len(csi_data.timestamps) and start_time < datetime.fromtimestamp(csi_data.timestamps[csi_i]) + delta_time < end_time:
                    print(row[0], row[3], start_time.isoformat(), end_time.isoformat(), (datetime.fromtimestamp(csi_data.timestamps[csi_i])).isoformat(), "save")
                    data.append([mesurement, row[0], row[3], start_time.isoformat(), end_time.isoformat(), (datetime.fromtimestamp(csi_data.timestamps[csi_i]) + delta_time).isoformat(), no_subcarriers, csi_matrix[csi_i].tolist()])
                    csi_i += 1
                mesurement += 1
    print("done")

    with open('output/output_5Ghz.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return data

`
def plot_csi_amplitudes(csi_amplitudes):
    """
    Plots the CSI amplitude heatmap and average amplitude per subcarrier.

    :param csi_amplitudes: CSI amplitude matrix (num_frames, num_subcarriers)
    """

    # Remove pilot subcarriers and whatever else is not data
    # data_subcarrier_indices = list(range(1, 27)) + list(range(38, 64))
    # csi_amplitudes = csi_amplitudes[:, data_subcarrier_indices]

    plt.figure(figsize=(12, 6))
    plt.imshow(csi_amplitudes.T, aspect="auto", cmap="jet", interpolation="nearest", origin="lower")
    plt.colorbar(label="CSI Amplitude (dB)")
    plt.xlabel("Frame Index (Time Progression)")
    plt.ylabel("Subcarrier Index")
    plt.title("CSI Amplitude Heatmap")
    plt.savefig(output_folder + "fig0")
    plt.show()

    csi_amplitudes = np.where(np.isinf(csi_amplitudes), np.nan, csi_amplitudes)

    mean_per_subcarrier = np.nanmean(csi_amplitudes, axis=0)
    print("Mean CSI Amplitude per Subcarrier:", mean_per_subcarrier)

    # plot mean amplitude per subcarrier
    plt.figure(figsize=(12, 6))
    plt.plot(mean_per_subcarrier, linestyle="-", marker="o", markersize=3)
    plt.xlabel("Subcarrier Index")
    plt.ylabel("Average CSI Amplitude (dB)")
    plt.title("Average CSI Amplitude per Subcarrier")
    plt.grid(True)
    plt.savefig(output_folder + "fig1")
    plt.show()

    avg_amplitude_per_frame = np.nanmean(csi_amplitudes, axis=1)  # Average over all subcarriers

    num_frames, num_subcarriers = csi_amplitudes.shape
    assert num_subcarriers <= 64
    group_size = 5

    num_groups = num_frames // group_size
    grouped_amplitudes = np.array([
        np.mean(avg_amplitude_per_frame[i * group_size: (i + 1) * group_size])
        for i in range(num_groups)
    ])

    x_values = np.arange(num_groups) * group_size

    # Plot the grouped average amplitude over time
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, grouped_amplitudes, linestyle='-')
    plt.xlabel("Frame Index (Grouped)")
    plt.ylabel("Average CSI Amplitude (dB)")
    plt.title(f"Average CSI Amplitude Over Time (Grouped by {group_size} Frames)")
    plt.grid(True)
    plt.savefig(output_folder + "fig2")
    plt.show()


def plot_amp():
    """
    CSIKIT Example: Plotting CSI Amplitude Heatmap
    (https://github.com/Gi-z/CSIKit)
    :return:
    """

    my_reader = get_reader(data_folder + filename)
    csi_data = my_reader.read_file(data_folder + filename, scaled=True)
    csi_matrix, no_frames, no_subcarriers = csitools.get_CSI(csi_data, metric="amplitude")

    # CSI matrix is now returned as (no_frames, no_subcarriers, no_rx_ant, no_tx_ant).
    # First we'll select the first Rx/Tx antenna pairing.
    csi_matrix_first = csi_matrix[:, :, 0, 0]
    # Then we'll squeeze it to remove the singleton dimensions.
    csi_matrix_squeezed = np.squeeze(csi_matrix_first)

    # This example assumes CSI data is sampled at ~100Hz.
    # In this example, we apply (sequentially):
    #  - a lowpass filter to isolate frequencies below 10Hz (order = 5)
    #  - a hampel filter to reduce high frequency noise (window size = 10, significance = 3)
    #  - a running mean filter for smoothing (window size = 10)

    sampling_rate = 40

    # for x in range(no_frames):
    #     csi_matrix_squeezed[x] = lowpass(csi_matrix_squeezed[x], 10, sampling_rate, 5)
    #     csi_matrix_squeezed[x] = hampel(csi_matrix_squeezed[x], 10, 3)
    #     csi_matrix_squeezed[x] = running_mean(csi_matrix_squeezed[x], 10)

    BatchGraph.plot_heatmap(csi_matrix_squeezed, csi_data.timestamps)

def csi_data_to_csv_mulithreaded(datafolder, timings_csv):
    person_names = []
    with open(timings_csv, mode='r') as file:
        csvreader = csv.reader(file)
        mesurement = 0
        saved_files = {}
        person = None
        for row in csvreader:
            person_name = row[0].lower()
            if person_name not in person_names:
                person_names.append(person_name)

        results = [None] * len(person_names)  # Placeholder for results in order

        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Submit all tasks and store futures
            future_to_index = {executor.submit(csi_data_to_csv, datafolder, timings_csv, person): idx
                               for idx, person in enumerate(person_names)}

            # Collect results in order
            for future in concurrent.futures.as_completed(future_to_index):
                print(f"DONE: {person_names[future]}")
                index = future_to_index[future]
                results[index] = future.result()

        concatineted_results = []
        for result in results:
            concatineted_results += result

        with open('output/output_5Ghz.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(concatineted_results)

        return results


#plot_csi_amplitudes(csi_amplitudes)
# plot_amp()

# data_folder = "./csi_collection/packet_data/"

data_folder = "unprocessed_data/locations"
timings_file = "../possition_timings.csv"

#csi_data = get_csi_data(filename)
csi_data_to_csv(data_folder, timings_file)

subprocess.run(["systemctl", "suspend"])

#print("CSI Amplitudes example:", csi_amplitudes)