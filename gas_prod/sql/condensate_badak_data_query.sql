SELECT prod_date AS date, realization AS condensate
FROM
f_lng_cond_prod_ptbadak('PT Badak','20130101', '20221110')
ORDER BY prod_date