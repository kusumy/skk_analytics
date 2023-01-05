SELECT prod_date AS date, realization AS condensate
FROM
f_lng_cond_prod_ptbadak('PT Badak','20221111', '20230430')
ORDER BY prod_date