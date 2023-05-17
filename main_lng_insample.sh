#!/bin/bash

# Activate the Anaconda Environment
. /root/anaconda3/bin/activate py38_ts

# Run every lng insample script
cd /opt/python-da-2022V2b/skk_analytics/lng/insample/
python feed_gas_tangguh_forecasting_insample.py
python lng_production_tangguh_forecasting_insample.py
python condensate_tangguh_forecasting_insample.py
python feed_gas_badak_forecasting_insample.py
python lng_production_badak_forecasting_insample.py
python condensate_badak_forecasting_insample.py
python c3_badak_forecasting_insample.py
python c4_badak_forecasting_insample.py

# deactivate the Anaconda environtment
. /root/anaconda3/bin/deactivate
