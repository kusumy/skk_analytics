#!/bin/bash

# Activate the Anaconda Environment
. /root/anaconda3/bin/activate py38_ts

# Run every lng insample script
cd /opt/python-da-2022V2b/skk_analytics/hse/forecasting/
python incident_rate_monthly_cum_forecasting.py
python incident_rate_yearly_forecasting.py

# Deactivate the Anaconda Environment
. /root/anaconda3/bin/deactivate