#!/bin/bash

# Activate the Anaconda Environment
. /root/anaconda3/bin/activate py38_ts

# Run every lng insample script
cd /opt/python-da-2022V2b/skk_analytics/lng/forecasting/
python feed_gas_tangguh_forecasting.py
python lng_production_tangguh_forecasting.py
python condensate_tangguh_forecasting.py
python feed_gas_badak_forecasting.py
python lng_prod_badak_forecasting.py
python condensate_badak_forecasting.py
python c3_badak_forecasting.py
python c4_badak_forecasting.py

# Deactivate the Anaconda Environment
. /root/anaconda3/bin/deactivate