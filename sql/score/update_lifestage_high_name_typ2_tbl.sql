update ?lifestage_high_name_typ2_tbl tgt
set tgt.eff_to_date = src.SYSTEMDATE -1, tgt.active=0
from (select * from ?score_output_tbl where systemdate=current_date()) src
where tgt.GROUP_LOYALTY_MEMBER_NUMBER = src.GROUP_LOYALTY_MEMBER_NUMBER
and tgt.LIFESTAGE_HIGH_NAME <> src.PRED_LIFESTAGE
and tgt.active=1