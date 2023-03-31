SELECT year_num, trir, survey_seismic, bor_eksplorasi, bor_eksploitasi, workover, wellservice
FROM trir_yearly
WHERE year_num BETWEEN '{}' AND '{}'
AND year_num <= (SELECT max(year_num) FROM trir_yearly WHERE trir is not null)
ORDER BY year_num