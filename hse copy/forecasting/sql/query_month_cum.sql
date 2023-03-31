select * from
(SELECT year_num, month_num, to_date(year_num || '-' || month_num || '-01', 'YYYY-MM-DD') AS datestamp, jam_kerja_cum, frequency_rate_cum, trir_cum, survey_seismic_cum,
bor_eksplorasi_cum, bor_eksploitasi_cum, workover_cum, wellservice_cum
FROM trir_monthly_cum ) b
where datestamp between '{}' and '{}'
order by datestamp