SELECT  year_num, month_num, jam_kerja_cum, frequency_rate_cum, trir_cum, survey_seismic_cum, 
        bor_eksplorasi_cum, bor_eksploitasi_cum, workover_cum, wellservice_cum
FROM    trir_monthly_cum
WHERE   year_num BETWEEN '{}' AND '{}'
AND year_num <= (SELECT max(year_num) FROM trir_monthly_cum WHERE trir_cum is not null)
ORDER BY year_num, month_num