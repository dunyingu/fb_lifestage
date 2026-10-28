merge into ?lifestage_typ2_tbl tgt
using (SELECT * FROM ?score_output_tbl
WHERE SYSTEMDATE = CURRENT_DATE() --date('2022-01-14') 
	  ) src
    on tgt.GROUP_LOYALTY_MEMBER_NUMBER = src.GROUP_LOYALTY_MEMBER_NUMBER

    when matched AND tgt.ACTIVE = 1 then 
        update set tgt.EFF_TO_DATE = src.SYSTEMDATE - 1, tgt.ACTIVE = 0