# Activate the Anaconda Environment
cd /home/spcuser/miniconda3/envs/py38_ts/bin
. activate py38_ts

# Run every lng insample script
cd /home/spcuser/Documents/code/python/skk/analytics/develop/skk_analytics/hse/forecasting
python incident_rate_monthly_cum_forecasting.py
python incident_rate_yearly_forecasting.py