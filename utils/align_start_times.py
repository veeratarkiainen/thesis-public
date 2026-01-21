import re, mne, os
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.simplefilter("ignore", category=RuntimeWarning)

BASE_DIR = r"C:\Users\Veera\Aalto\Thesis"
os.chdir(BASE_DIR)

# Go through rows in subject list text file
file_names=[]
with open('data/subject_list.txt') as subjects:
    for line in subjects:
        # Get subject number
        line = line.strip()
        line_data = line.split("_")
        sub = line_data[1]
        date = line_data[0]
        # Load EEG data
        eeg_file = os.path.join("data", date + "_" + sub, date + "_" + sub + ".vhdr")
        file_names.append(eeg_file)

# Define the folder containing the PsychoPy data
folder_path = "data/PsychoPy"

# Function to extract subject number from filename
def extract_subject_number(filename):
    match = re.search(r"subject_(\d+)", filename)
    return int(match.group(1)) #if match else float('inf')

# List all CSV files and sort them numerically by subject number
csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')],
                   key=extract_subject_number)

psychopy_times = []

for file in csv_files:
    file_path = os.path.join(folder_path, file)

    try:
        # Load the CSV file
        df = pd.read_csv(file_path)

        # Ensure 'expStart' exists and is not empty
        if "expStart" in df.columns:
            raw_time = df["expStart"].dropna().iloc[0]  # Get the first non-null value
            raw_time = str(raw_time).strip()  # Remove extra spaces or newlines

            try:
                # Try parsing automatically
                parsed_time = pd.to_datetime(raw_time, errors='coerce')

                # Manual parsing fails
                if pd.isna(parsed_time):
                    parsed_time = pd.to_datetime(raw_time, format="%Y-%m-%d %Hh%M.%S.%f %z", errors='coerce')

                formatted_time = parsed_time.strftime("%H:%M:%S")  # Extract only time
                psychopy_times.append(formatted_time)  # Store in list

            except Exception as e:
                print(f"{file}: Error parsing time '{raw_time}': {e}")

    except Exception as e:
        print(f"Error processing {file}: {e}")


def read_raw_data(file_names):

    sub_names = ["Sub0" + str(i) for i in range(len(file_names))]
    measurement_times = []

    for i in range(len(sub_names)):
        raw = mne.io.read_raw_brainvision(file_names[i], preload=True, verbose=False)
        measurement_time = raw.info["meas_date"]
        measurement_times.append(measurement_time.strftime("%H:%M:%S"))

    return measurement_times

measurement_times = read_raw_data(file_names)

print("start times according to psychopy", psychopy_times)
print("start times according to measurements", measurement_times)

# Plan B: go through manually for one subject and generalize
reference_6 = measurement_times[6] # Reference time of one subject
original_6 = psychopy_times[6]  # Original time of one subject
time_format = "%H:%M:%S"  # Time format

# Convert strings to datetime objects
time_obj = datetime.strptime(reference_6, time_format)
time_obj_pp = datetime.strptime(original_6, time_format)

# Subtract 5 seconds from absolute_6
adjusted_time_obj = time_obj - timedelta(seconds=5) # 5-second relative difference, from Logfiles

# Compute the absolute difference
absolute_difference = adjusted_time_obj - time_obj_pp

# Extract minutes and seconds
minutes, seconds = divmod(absolute_difference.total_seconds(), 60)
print(f"Absolute difference: {int(minutes)} minutes, {int(seconds)} seconds") # is always the same

absolute_offset = timedelta(minutes=23, seconds=15)

# Convert times and compute relative differences
relative_differences = []
for psychopy_time, meas_time in zip(psychopy_times, measurement_times):
    # Convert to datetime
    psychopy_dt = datetime.strptime(psychopy_time, time_format)
    meas_dt = datetime.strptime(meas_time, time_format)

    # Adjust measurement time by removing absolute offset
    adjusted_meas_dt = meas_dt - absolute_offset

    # Compute relative difference
    relative_difference = adjusted_meas_dt - psychopy_dt
    minutes, seconds = divmod(relative_difference.total_seconds(), 60)

    # Store results
    relative_differences.append((int(minutes), int(seconds)))

relative_differences_seconds = [m * 60 + s for m, s in relative_differences] #
relative_differences_seconds[0] += 20 # manual tweaking for one subject

print("Relative differences in seconds", relative_differences_seconds)