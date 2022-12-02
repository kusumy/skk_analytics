SELECT prod_date AS date, realization_prod AS lng_production, fg_exogenous AS fg_exog
FROM f_lng_prod_ptbadak('PT Badak','20130101', '20221110')
WHERE prod_date BETWEEN '2013-01-01' AND '2022-11-10'
ORDER BY prod_date asc