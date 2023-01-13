SELECT  year_num, month_num, jam_kerja_cum, frequency_rate_cum, trir_cum, survey_seismic_cum, 
        bor_eksplorasi_cum, bor_eksploitasi_cum, workover_cum, wellservice_cum
FROM    trir_monthly_cum
WHERE   year_num BETWEEN '2013' AND date_part('year', now())
ORDER BY year_num, month_num