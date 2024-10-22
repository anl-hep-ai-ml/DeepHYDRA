#!/bin/bash

json_dir="informers/results"
csv_dir="data_plotting"
python_script="download_run_df.py"



missing_runs=()

# Iterate over the JSON files
for json_file in "$json_dir"/run_*.json; 
do
  # Extract the run number from the JSON file name
  run_number=$(basename "$json_file" | sed 's/run_\([0-9]*\)\.json/\1/')

  # Check if the corresponding CSV file exists
  csv_file="$csv_dir/hlt_data_pd_$run_number.csv"
  if [[ ! -f "$csv_file" ]]; then
    echo "CSV file for run number $run_number not found."
    missing_runs+=("$run_number")
  else
    echo "CSV file for run number $run_number exists."
  fi
done

# Check if there are any missing runs
if [ ${#missing_runs[@]} -ne 0 ]; then
  # Convert the list of missing runs to a comma-separated string
  missing_runs_str=$(IFS=, ; echo "${missing_runs[*]}")

  # Run the Python script with the list of missing runs as an argument
  echo "Missing CSV files for run numbers: $missing_runs_str"
  python3 "$python_script" --run_numbers "$missing_runs_str" --anomaly_json_dir "$json_dir"
else
  echo "All CSV files are present."
fi
