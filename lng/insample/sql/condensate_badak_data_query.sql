SELECT prod_date AS date, realization_prod AS condensate
FROM f_lng_cond_prod_ptbadak('PT Badak','{}', '{}')
WHERE prod_date <= (SELECT max(prod_date) FROM f_lng_cond_prod_ptbadak('BP Tangguh', '20130101', current_date-2) WHERE realization_prod is not null)
ORDER BY prod_date