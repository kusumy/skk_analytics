SELECT prod_date AS date, fg_exogenous AS fg_exog
FROM f_lng_prod_ptbadak('PT Badak','20130101', '20221110')
WHERE prod_date BETWEEN '{}' AND '{}'
ORDER BY prod_date
