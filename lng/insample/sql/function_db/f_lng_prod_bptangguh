CREATE OR REPLACE FUNCTION public.f_lng_prod_bptangguh(lngplant character varying, startdate date, enddate date)
 RETURNS TABLE(lng_plant character varying, prod_date date, realization_prod double precision, is_unplanned_shutdown integer, is_planned_shutdown integer)
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
		_query = _query || 'prod.prod_date, prod.realization ';
		if(_upsCount > 0)
			then _query = _query || ', coalesce(ups.is_unplanned_shutdown,0) as is_unplanned_shutdown ';
			else _query = _query || ', 0 as is_unplanned_shutdown';
		end if;
		if(_psCount > 0)
			then _query = _query || ', coalesce(ps.is_planned_shutdown,0) as is_planned_shutdown ';
			else _query = _query || ', 0 as is_unplanned_shutdown';
		end if;
		_query = _query || ' from lng_production_daily as prod ';
		if(length(_query_union_ups)::int > 0)
		then
			_query_union_ups = substr(_query_union_ups, 1, length(_query_union_ups) - 7);
			_query = _query || 'left join ( ';
			_query = _query || _query_union_ups;
			_query = _query || ') ups ';
			_query = _query || ' on prod.prod_date = ups.unplanned_series ';
		end if;
		if(length(_query_union_ps)::int > 0)
		then
			_query_union_ps = substr(_query_union_ps, 1, length(_query_union_ps) - 7);
			_query = _query || 'left join ( ';
			_query = _query || _query_union_ps;
			_query = _query || ') ps ';
			_query = _query || ' on prod.prod_date = ps.planned_series ';
		end if;
		_query = _query || ' where prod.lng_plant = ''' || _lng_plant || '''';
		_query = _query || '  and prod.prod_date between ''' || to_char(startdate,'YYYYMMDD');
		_query = _query || ''' and ''' || to_char(enddate,'YYYYMMDD') || '''';
		_query = _query || ' order by prod.prod_date';
		--raise notice '%', _query;
		return QUERY 
		execute _query;
	end;
$function$
;
