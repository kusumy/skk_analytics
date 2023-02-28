SELECT prod_date AS date, realization_prod AS lng_production, fg_exogenous AS fg_exog
FROM f_lng_prod_ptbadak('PT Badak','20130101', '20231231')
WHERE prod_date BETWEEN '{}' AND '{}'
AND prod_date <= (SELECT max(prod_date) FROM f_lng_prod_ptbadak('PT Badak','20130101', '20231231') WHERE realization_prod is not null and lng_plant = 'PT Badak')
ORDER BY prod_date