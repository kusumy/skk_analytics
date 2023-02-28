SELECT prod_date AS date, realization_prod AS condensate
FROM f_lng_cond_prod_ptbadak('PT Badak','{}', '{}')
ORDER BY prod_date