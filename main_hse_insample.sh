# Activate the Anaconda Environment
cd /home/spcuser/miniconda3/envs/py38_ts/bin
. activate py38_ts

# Run every lng insample script
cd /home/spcuser/Documents/code/python/skk/analytics/develop/skk_analytics/hse/insample
python incident_rate_monthly_cumulative_insample.py
python yearly_incident_rate_insample.py