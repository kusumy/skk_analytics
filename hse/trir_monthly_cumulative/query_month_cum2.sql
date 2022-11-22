select year, date_part('month', "MONTH") as month_num, 
"JAM_KERJA", "JAM_KERJA_CUM",
"FREQUENCY_RATE", "FREQUENCY_RATE_CUM",
ROUND(("FREQUENCY_RATE_CUM" * 1000000 / "JAM_KERJA_CUM"),2) as TRIR_CUM,
"SURVEI_SEISMIC", "BOR_EKSPLORASI", "BOR_EKSPLOITASI", "WORKOVER", "WELLSERVICE"
from (
        select year, "MONTH", max("JAM_KERJA") as "JAM_KERJA", max("JAM_KERJA_CUM") as "JAM_KERJA_CUM",
        (coalesce(MAX("FATAL"),0) + coalesce(max("BERAT"),0) + coalesce(max("SEDANG"),0) + coalesce(max("RINGAN"),0) 
        + coalesce(max("FATALITY"),0) + coalesce(max("LWDC"),0) + coalesce(max("RWDC"),0) + coalesce(max("MTC"),0)) 
        as "FREQUENCY_RATE",
        (coalesce(MAX("FATAL_CUM"),0) + coalesce(max("BERAT_CUM"),0) + coalesce(max("SEDANG_CUM"),0) + coalesce(max("RINGAN_CUM"),0) 
        + coalesce(max("FATALITY_CUM"),0) + coalesce(max("LWDC_CUM"),0) + coalesce(max("RWDC_CUM"),0) + coalesce(max("MTC_CUM"),0)) 
        as "FREQUENCY_RATE_CUM", coalesce(MAX("SURVEI_SEISMIC"),0) as "SURVEI_SEISMIC",
        coalesce(MAX("BOR_EKSPLORASI"),0) as "BOR_EKSPLORASI", coalesce(MAX("BOR_EKSPLOITASI"),0) as "BOR_EKSPLOITASI", 
        coalesce(MAX("WORKOVER"),0) as "WORKOVER", coalesce(MAX("WELLSERVICE"),0) as "WELLSERVICE"
        from
        (
            select date_part('year', "MONTH" ) as year, "MONTH", 
            sum("JAM_KERJA") over (partition by "MONTH",  date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "JAM_KERJA",
            SUM("JAM_KERJA") over (partition by date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH" ) as "JAM_KERJA_CUM",
            sum("FATAL")  over (partition by "MONTH",  date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "FATAL",
            sum("BERAT")  over (partition by "MONTH",  date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "BERAT",
            sum("SEDANG")  over (partition by "MONTH",  date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "SEDANG",
            sum("RINGAN")  over (partition by "MONTH",  date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "RINGAN",
            sum("LWDC")  over (partition by "MONTH",  date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "FATALITY",
            sum("LWDC")  over (partition by "MONTH",  date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "LWDC",
            sum("RWDC")  over (partition by "MONTH",  date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "RWDC",
            sum("MTC")  over (partition by "MONTH",  date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "MTC",
            sum("FATAL")  over (partition by date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "FATAL_CUM",
            sum("BERAT")  over (partition by date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "BERAT_CUM",
            sum("SEDANG")  over (partition by date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "SEDANG_CUM",
            sum("RINGAN")  over (partition by date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "RINGAN_CUM",
            sum("FATALITY")  over (partition by date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "FATALITY_CUM",
            sum("LWDC")  over (partition by date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "LWDC_CUM",
            sum("RWDC")  over (partition by date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "RWDC_CUM",
            sum("MTC")  over (partition by date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "MTC_CUM",
            sum("SURVEI_SEISMIC") over (partition by "MONTH",  date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "SURVEI_SEISMIC",
            sum("BOR_EKSPLORASI") over (partition by "MONTH",  date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "BOR_EKSPLORASI",
            sum("BOR_EKSPLOITASI") over (partition by "MONTH",  date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "BOR_EKSPLOITASI",
            sum("WORKOVER") over (partition by "MONTH",  date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "WORKOVER",
            sum("WELLSERVICE") over (partition by "MONTH",  date_part('year', "MONTH" ) order by date_part('year', "MONTH" ), "MONTH") as "WELLSERVICE"
            from hse_jam_kerja 
            order by "MONTH" 
        ) a
    group by year,"MONTH"
    order by year,"MONTH"
    ) b