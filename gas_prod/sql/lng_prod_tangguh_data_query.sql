select prod_date as date, realization_prod as lng_production, is_planned_shutdown as planned_shutdown, is_unplanned_shutdown as unplanned_shutdown
FROM f_lng_prod_bptangguh('BP Tangguh', '20160101', '20220914')