# Activate the Anaconda Environment
cd /home/spcuser/miniconda3/envs/py38_ts/bin
. activate py38_ts

# Run every lng insample script
cd /home/spcuser/Documents/code/python/skk/analytics/develop/skk_analytics/lng/forecasting
python feed_gas_tangguh_forecasting.py
python lng_production_tangguh_forecasting.py
python condensate_tangguh_forecasting.py
python feed_gas_badak_forecasting.py
python lng_prod_badak_forecasting.py
python condensate_badak_forecasting.py
python c3_badak_forecasting.py
python c4_badak_forecasting.py
