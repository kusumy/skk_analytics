CREATE OR REPLACE FUNCTION public.f_lng_prod_ptbadak_grf_only(lngplant character varying, startdate date, enddate date)
 RETURNS TABLE(lng_plant character varying, prod_date date, realization_prod double precision, fg_exogenous double precision, is_unplanned_shutdown integer, is_planned_shutdown integer)
 LANGUAGE plpgsql
AS $function$ 
	#variable_conflict use_column
	declare _unplanned_shutdown record;
	declare _planned_shutdown record;
	declare _lng_plant varchar;
	declare _query varchar;
	declare _query_union_ups varchar;
	declare _query_union_ps varchar;
	declare _upsCount int;
	declare _psCount int;
	begin
		_lng_plant = lngplant;
		_query_union_ups = '';
		_query_union_ps = '';
		_upsCount = 0;
		_psCount = 0;
		for _unplanned_shutdown in 
			select lng_plant, start_date, end_date, note
			from lng_unplanned_shutdown where lng_plant = _lng_plant
		loop
			_upsCount = _upsCount + 1;
			_query_union_ups = _query_union_ups || 'select generate_series(date ''' ||  to_char(_unplanned_shutdown.start_date,'YYYYMMDD')  || ''',';
			_query_union_ups = _query_union_ups || 'date ''' ||  to_char(_unplanned_shutdown.end_date,'YYYYMMDD')  || ''',';
			_query_union_ups = _query_union_ups || '''1 day'') as unplanned_series, 1 as is_unplanned_shutdown union ';
		end loop;
		for _planned_shutdown in 
			select lng_plant, start_date, end_date, note
			from lng_planned_shutdown where lng_plant = _lng_plant
		loop
			_psCount = _psCount + 1;
			_query_union_ps = _query_union_ps || 'select generate_series(date ''' ||  to_char(_planned_shutdown.start_date,'YYYYMMDD')  || ''',';
			_query_union_ps = _query_union_ps || 'date ''' ||  to_char(_planned_shutdown.end_date,'YYYYMMDD')  || ''',';
			_query_union_ps = _query_union_ps || '''1 day'') as planned_series, 1 as is_planned_shutdown union ';
		end loop;
		_query = 'select ''' || _lng_plant || '''::varchar as lng_plant, ';
		_query = _query || 'lng.prod_date, lng.realization, ';
		_query = _query || '  CASE WHEN grf.grf IS NOT NULL THEN grf.grf ';
		_query = _query || '  WHEN fg.realization IS NOT NULL THEN fg.realization ';
		_query = _query || '  WHEN grf_lastyear.grf IS NOT NULL THEN grf_lastyear.grf ';
		_query = _query || '  ELSE NULL::double precision ';
		_query = _query || 'END AS fg_exogenous';
		if(_upsCount > 0)
			then _query = _query || ', coalesce(ups.is_unplanned_shutdown,0) as is_unplanned_shutdown ';
			else _query = _query || ', 0 as is_unplanned_shutdown';
		end if;
		if(_psCount > 0)
			then _query = _query || ', coalesce(ps.is_planned_shutdown,0) as is_planned_shutdown ';
			else _query = _query || ', 0 as is_unplanned_shutdown';
		end if; 
		_query = _query || ' from lng_production_daily lng ';
		if(length(_query_union_ups)::int > 0)
		then
			_query_union_ups = substr(_query_union_ups, 1, length(_query_union_ups) - 7);
			_query = _query || 'left join ( ';
			_query = _query || _query_union_ups;
			_query = _query || ') ups ';
			_query = _query || ' on lng.prod_date = ups.unplanned_series ';
		end if;
		if(length(_query_union_ps)::int > 0)
		then
			_query_union_ps = substr(_query_union_ps, 1, length(_query_union_ps) - 7);
			_query = _query || 'left join ( ';
			_query = _query || _query_union_ps;
			_query = _query || ') ps ';
			_query = _query || ' on lng.prod_date = ps.planned_series ';
		end if;
		_query = _query || ' LEFT JOIN lng_target_badak_daily_feed_gas_grf grf ON lng.prod_date = grf.date';
		_query = _query || ' LEFT JOIN lng_target_badak_adp_monthly adp ON date_trunc(''MONTH'', lng.prod_date) = adp.prod_date';
		_query = _query || ' LEFT JOIN lng_feed_gas_daily fg ON lng.prod_date = fg.prod_date AND fg.lng_plant =''PT Badak''';
		_query = _query || ' LEFT JOIN ( SELECT lng_target_badak_daily_feed_gas_grf.date,';
		_query = _query || '   lng_target_badak_daily_feed_gas_grf.date + ''1 year''::interval AS last_year_date,';
		_query = _query || '   lng_target_badak_daily_feed_gas_grf.grf';
		_query = _query || '   FROM lng_target_badak_daily_feed_gas_grf) grf_lastyear ON lng.prod_date = grf_lastyear.last_year_date';
		_query = _query || ' WHERE lng.lng_plant::text = ''PT Badak'''; 
		_query = _query || ' and lng.prod_date between ''' || startdate || ''' and ''' || enddate || ''''; 
		_query = _query || ' ORDER BY lng.prod_date DESC';
		raise notice '%', _query;
		return QUERY 
		execute _query;
	end;
$function$
;
