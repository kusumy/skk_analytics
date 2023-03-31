select * from
(
select year_num, month_num, to_date(year_num || '-' || month_num || '-01', 'YYYY-MM-DD') AS datestamp,
max(survei_seismic) as survei_seismic,
max(survei_seismic_cum) as survei_seismic_cum,
max(drilling_explor) as drilling_explor,
max(drilling_explor_cum) as drilling_explor_cum,
max(drilling_explot) as drilling_explot,
max(drilling_explot_cum) as drilling_explot_cum,
max(workover) as workover,
max(workover_cum) as workover_cum,
max(wellservice) as wellservice,
max(wellservice_cum) as wellservice_cum
from(
select survei.year_num, survei.month_num, 
sum(total_survey) over (partition by survei.month_num, survei.year_num  order by survei.year_num , survei.month_num) as survei_seismic,
sum(total_survey) over (partition by survei.year_num  order by survei.year_num , survei.month_num) as survei_seismic_cum,
sum(drilling_explor) over (partition by survei.month_num, survei.year_num  order by survei.year_num , survei.month_num) as drilling_explor,
sum(drilling_explor) over (partition by survei.year_num  order by survei.year_num , survei.month_num) as drilling_explor_cum,
sum(drilling_explot) over (partition by survei.month_num, survei.year_num  order by survei.year_num , survei.month_num) as drilling_explot,
sum(drilling_explot) over (partition by survei.year_num  order by survei.year_num , survei.month_num) as drilling_explot_cum,
sum(workover) over (partition by survei.month_num, survei.year_num  order by survei.year_num , survei.month_num) as workover ,
sum(workover) over (partition by survei.year_num  order by survei.year_num , survei.month_num) as workover_cum,
sum(wellservice) over (partition by survei.month_num, survei.year_num  order by survei.year_num , survei.month_num) as wellservice ,
sum(wellservice) over (partition by survei.year_num  order by survei.year_num , survei.month_num) as wellservice_cum
from drilling_survey_bln survei
left join
(
select year_num, month_num, sum(rencana_bulanan_wp_b) as drilling_explor
from drilling_explor_bln group by year_num, month_num 
) explor on survei.year_num = explor.year_num and survei.month_num = explor.month_num 
left join
(
select year_num, month_num, sum(rencana_bulanan_wp_b) as drilling_explot
from drilling_explot_bln group by year_num, month_num 
) explot on survei.year_num = explot.year_num and survei.month_num = explot.month_num 
left join
(
select year_num, month_num, sum(rencana_bulanan_wp_b) as workover 
from drilling_ku_perbln group by year_num, month_num
) ku on survei.year_num = ku.year_num and survei.month_num = ku.month_num
left join
(
select year_num, month_num, sum(rencana_bulanan_wp_b) as wellservice 
from drilling_ps_perbln dpp group by year_num, month_num
) ps on survei.year_num = ps.year_num and survei.month_num = ps.month_num
) drilling
group by year_num, month_num
) as a
where a.datestamp between '{}' and '{}'
order by a.year_num, a.month_num