# import h5py
#
# def explore_hdf5_group(group, indent=""):
#     """Explore an HDF5 group recursively and print its contents."""
#     for name, item in group.items():
#         if isinstance(item, h5py.Group):  # If the item is another group
#             print(f"{indent}Group: {name}")
#             explore_hdf5_group(item, indent + "  ")
#         elif isinstance(item, h5py.Dataset):  # If the item is a dataset
#             print(f"{indent}Dataset: {name}")
#             print(f"{indent}  Shape: {item.shape}")
#             print(f"{indent}  Dtype: {item.dtype}")
#             # If you want to print the actual data (might be very long!)
#             # print(f"{indent}  Data: {item[:]}")
#         else:
#             print(f"{indent}Unknown type: {name}")
#
# def explore_hdf5_file(file_name):
#     """Explore the entire HDF5 file."""
#     with h5py.File(file_name, 'r') as f:
#         print(f"File: {file_name}")
#         explore_hdf5_group(f)
#
# if __name__ == "__main__":
#     # hdf5_file_name = "data/experiments/external_data/2022-06-07_experiment_S00/2022-06-07_18-11-00_actionNet-microphone_S00/2022-06-07_18-11-04_streamLog_actionNet-microphone_S00.hdf5"  # Replace with your file path
#     hdf5_file_name = "data/experiments/wearable_data/2022-06-07_experiment_S00/2022-06-07_17-18-17_actionNet-wearables_S00/2022-06-07_17-18-46_streamLog_actionNet-wearables_S00.hdf5"
#     explore_hdf5_file(hdf5_file_name)


import h5py

def extract_time_range_from_hdf5(file_path, time_dataset_path):
    """Extract the time range from the specified dataset in an HDF5 file."""
    with h5py.File(file_path, 'r') as f:
        time_data = f[time_dataset_path][:]
        start_time = time_data.min()
        end_time = time_data.max()
    return start_time, end_time

def are_time_ranges_overlapping(range1, range2):
    """Check if two time ranges overlap."""
    return range1[0] <= range2[1] and range2[0] <= range1[1]

if __name__ == "__main__":
    # Paths to HDF5 files and time datasets
    microphone_file = "data/experiments/external_data/2022-06-07_experiment_S00/2022-06-07_17-13-37_actionNet-microphone_S00/2022-06-07_17-13-41_streamLog_actionNet-microphone_S00.hdf5"
    overhead_time_path = "overhead/chunk_timestamp/time_s"
    sink_time_path = "sink/chunk_timestamp/time_s"

    label_file = "data/experiments/wearable_data/2022-06-07_experiment_S00/2022-06-07_17-18-17_actionNet-wearables_S00/2022-06-07_17-18-46_streamLog_actionNet-wearables_S00.hdf5"
    label_time_path = "experiment-activities/activities/time_s"

    overhead_range = extract_time_range_from_hdf5(microphone_file, overhead_time_path)
    sink_range = extract_time_range_from_hdf5(microphone_file, sink_time_path)
    label_range = extract_time_range_from_hdf5(label_file, label_time_path)

    if are_time_ranges_overlapping(overhead_range, label_range) or are_time_ranges_overlapping(sink_range, label_range):
        print("The labels might correspond to the microphone data.")
    else:
        print("The labels do not seem to correspond to the microphone data.")