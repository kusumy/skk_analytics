#!/bin/bash

# Activate the Anaconda Environment
. /root/anaconda3/bin/activate py38_ts

# Run every lng insample script
cd /opt/python-da-2022V2b/skk_analytics/hse/insample/
python incident_rate_monthly_cumulative_insample.py
python yearly_incident_rate_insample.py

# Deactivate the Anaconda Environment
. /root/anaconda3/bin/deactivate