with current_score_date as 
(select max(eff_from_date) as cur_score_date from ?lifestage_typ2_tbl
)
,
last_score_date as
(select max(eff_from_date) as last_score_dat
from ?lifestage_typ2_tbl
where eff_from_date < (select cur_score_date from current_score_date)
)
,
mbr_score_hist as
(
select distinct
GROUP_LOYALTY_MEMBER_NUMBER,
EFF_FROM_DATE as score_date,
CONCAT(LIFESTAGE_HIGH_NAME_LABEL,'.',LIFESTAGE_HIGH_NAME) as lifestage_segment
from ?lifestage_typ2_tbl
where score_date >= (select last_score_dat from last_score_date)
-- order by GROUP_LOYALTY_MEMBER_NUMBER, score_date
)


SELECT * FROM 
(
select 
GROUP_LOYALTY_MEMBER_NUMBER,
score_date as previous_score_date,
lifestage_segment as previous_lifestage_segment,
lead(score_date) over (partition by GROUP_LOYALTY_MEMBER_NUMBER order by score_date) AS current_score_date,
lead(lifestage_segment) over (partition by GROUP_LOYALTY_MEMBER_NUMBER order by score_date) AS current_lifestage_segment
from mbr_score_hist
)
--WHERE current_score_date IS NOT NULL 
--and GROUP_LOYALTY_MEMBER_NUMBER in (
--select GROUP_LOYALTY_MEMBER_NUMBER
--from mbr_score_hist
--group by 1
--having count(*) >1
--)
