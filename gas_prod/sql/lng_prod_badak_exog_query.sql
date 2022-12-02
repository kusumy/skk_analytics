SELECT prod_date AS date, fg_exogenous AS fg_exog
FROM f_lng_prod_ptbadak('PT Badak','20130101', '20221110')
WHERE prod_date BETWEEN '2022-11-11' AND '2022-12-31'
ORDER BY prod_date asc