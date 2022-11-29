SELECT year_num, trir, survey_seismic, bor_eksplorasi, bor_eksploitasi, workover, wellservice
FROM trir_yearly
WHERE
year_num BETWEEN '2013' AND '2022'