#!/bin/bash

# Access the path to the directory config
config_file="/home/spcuser/Documents/code/python/skk/analytics/develop/skk_analytics/rootdirectory.ini"
conda_dir_value=$(grep "conda_dir" "$config_file" | cut -d "=" -f 2 | tr -d '[:space:]')
home_dir_value=$(grep "home_dir" "$config_file" | cut -d "=" -f 2 | tr -d '[:space:]')

# Activate the Anaconda Environment
. "$conda_dir_value/activate" py38_ts

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time"

# Run every lng insample script
cd "$home_dir_value/lng/insample"
python feed_gas_tangguh_forecasting_insample.py
python lng_production_tangguh_forecasting_insample.py
python condensate_tangguh_forecasting_insample.py
python feed_gas_badak_forecasting_insample.py
python lng_production_badak_forecasting_insample.py
python condensate_badak_forecasting_insample.py
python c3_badak_forecasting_insample.py
python c4_badak_forecasting_insample.py

end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script ended at: $end_time"

start_seconds=$(date -d "$start_time" +%s)
end_seconds=$(date -d "$end_time" +%s)
duration_seconds=$((end_seconds - start_seconds))

duration_hours=$((duration_seconds / 3600))
duration_minutes=$(((duration_seconds % 3600) / 60))
duration_seconds=$((duration_seconds % 60))

echo "Duration: $duration_hours hours, $duration_minutes minutes, $duration_seconds seconds"

log_directory="$home_dir_value/lng/insample/logs"

# Redirect output to log file
echo "Start Time: $start_time" >> "$log_directory/executing_main_lng_insample_log.txt"
echo "End Time: $end_time" >> "$log_directory/executing_main_lng_insample_log.txt"
echo "Duration: $duration_hours hours, $duration_minutes minutes, $duration_seconds seconds" >> "$log_directory/executing_main_lng_insample_log.txt"

# deactivate the Anaconda environtment
. "$conda_dir_value/deactivate"
