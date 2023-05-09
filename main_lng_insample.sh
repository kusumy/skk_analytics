# Activate the Anaconda Environment
cd /home/spcuser/miniconda3/envs/py38_ts/bin
. activate py38_ts

# Run every lng insample script
cd /home/spcuser/Documents/code/python/skk/analytics/develop/skk_analytics/lng/insample
python feed_gas_tangguh_forecasting_insample.py
python lng_production_tangguh_forecasting_insample.py
python condensate_tangguh_forecasting_insample.py
python feed_gas_badak_forecasting_insample.py
python lng_production_badak_forecasting_insample.py
python condensate_badak_forecasting_insample.py
python c3_badak_forecasting_insample.py
python c4_badak_forecasting_insample.py
