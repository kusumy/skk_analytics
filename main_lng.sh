#!/bin/bash

# Access the path to the directory config
config_file="/home/spcuser/Documents/code/python/skk/analytics/develop/skk_analytics/config/rootdirectory.ini"
conda_dir_value=$(grep "conda_dir" "$config_file" | cut -d "=" -f 2 | tr -d '[:space:]')
home_dir_value=$(grep "home_dir" "$config_file" | cut -d "=" -f 2 | tr -d '[:space:]')

# Activate the Anaconda Environment
. "$conda_dir_value/activate" py38_ts

start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script started at: $start_time"

# Run every lng insample script
cd "$home_dir_value/lng/forecasting"
#python feed_gas_tangguh_forecasting.py
#python lng_production_tangguh_forecasting.py
#python condensate_tangguh_forecasting.py
#python feed_gas_badak_forecasting.py
#python lng_prod_badak_forecasting.py
#python condensate_badak_forecasting.py
#python c3_badak_forecasting.py
python c4_badak_forecasting.py

end_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Script ended at: $end_time"

start_seconds=$(date -d "$start_time" +%s)
end_seconds=$(date -d "$end_time" +%s)
duration_seconds=$((end_seconds - start_seconds))

duration_hours=$((duration_seconds / 3600))
duration_minutes=$(((duration_seconds % 3600) / 60))
duration_seconds=$((duration_seconds % 60))

echo "Duration: $duration_hours hours, $duration_minutes minutes, $duration_seconds seconds"

log_directory="$home_dir_value/lng/forecasting/logs"

# Redirect output to log file
echo "Start Time: $start_time" >> "$log_directory/executing_main_lng_log.txt"
echo "End Time: $end_time" >> "$log_directory/executing_main_lng_log.txt"
echo "Duration: $duration_hours hours, $duration_minutes minutes, $duration_seconds seconds" >> "$log_directory/executing_main_lng_log.txt"

# Deactivate the Anaconda Environment
. "$conda_dir_value/deactivate"
