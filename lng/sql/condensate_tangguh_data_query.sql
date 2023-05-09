SELECT prod_date AS date, realization_prod AS condensate, target_wpnb AS wpnb_oil, is_unplanned_shutdown AS unplanned_shutdown, is_planned_shutdown AS planned_shutdown
FROM f_lng_condensate_bptangguh('BP Tangguh', '{}', '{}')
WHERE prod_date <= (SELECT max(prod_date) FROM f_lng_condensate_bptangguh('BP Tangguh', '20160101', current_date-2) WHERE realization_prod is not null)
ORDER BY prod_date