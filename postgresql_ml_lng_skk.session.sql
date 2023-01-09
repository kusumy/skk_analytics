SELECT running_date, model_param_c
FROM lng_analytics_model_param 
WHERE lng_plant = 'PT Badak' 
AND product = 'LNG Production'
ORDER BY running_date DESC 
LIMIT 1 OFFSET 0


query data sampai H-1
data hanya sampai H-4

LNG & Feed & Condensate
testing = (H-4) - 365
training = data - testing

h-4 = 1
h-3 = 1
h-2 = 1