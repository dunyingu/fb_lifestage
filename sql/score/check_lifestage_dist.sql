with lifestage_seg_cnt_current AS (
select 
concat(PRED_LABEL, '.',PRED_LIFESTAGE) as lifestage_segment,
avg(MBR_AGE) as avg_member_age_current,
count(distinct GROUP_LOYALTY_MEMBER_NUMBER) as TOTAL_MEMBER_CNT_current
from ?score_output_tbl
where SYSTEMDATE = (select max(SYSTEMDATE) from ?score_output_tbl)
GROUP BY 1
)
,
lifestage_seg_prob_mean_current as 
(select
avg(_0_SINGLE_PROB) as _0_SINGLE_PROB_MEAN,
avg(_1_COUPLE_NO_KIDS_PROB) as _1_COUPLE_NO_KIDS_PROB_MEAN,
avg(_2_FAMILY_W_BABY_PROB) as _2_FAMILY_W_BABY_PROB_MEAN,
avg(_3_FAMILY_W_CHILDREN_PROB) as _3_FAMILY_W_CHILDREN_PROB_MEAN,
avg(_4_FAMILY_W_BABY_OTHER_CHILDREN_PROB) as _4_FAMILY_W_BABY_OTHER_CHILDREN_PROB_MEAN,
avg(_5_MULTI_ADULT_FAMILY_PROB) as _5_MULTI_ADULT_FAMILY_PROB_MEAN
from ?score_output_tbl
where SYSTEMDATE = (select max(SYSTEMDATE) from ?score_output_tbl)
)
,
lifestage_seg_prob_mean_current_row as (
select '0.SINGLE' as lifestage_segment, _0_SINGLE_PROB_MEAN as AVG_PROB_current from lifestage_seg_prob_mean_current
union
select '1.COUPLE_NO_KIDS' as lifestage_segment, _1_COUPLE_NO_KIDS_PROB_MEAN as AVG_PROB_current from lifestage_seg_prob_mean_current
union
select '2.FAMILY_W_BABY' as lifestage_segment, _2_FAMILY_W_BABY_PROB_MEAN as AVG_PROB_current from lifestage_seg_prob_mean_current
union
select '3.FAMILY_W_CHILDREN' as lifestage_segment, _3_FAMILY_W_CHILDREN_PROB_MEAN as AVG_PROB_current from lifestage_seg_prob_mean_current
union
select '4.FAMILY_W_BABY_OTHER_CHILDREN' as lifestage_segment, _4_FAMILY_W_BABY_OTHER_CHILDREN_PROB_MEAN as AVG_PROB_current from lifestage_seg_prob_mean_current
union
select '5.MULTI-ADULT_FAMILY' as lifestage_segment,_5_MULTI_ADULT_FAMILY_PROB_MEAN as AVG_PROB_current from lifestage_seg_prob_mean_current
)
,
last_score_date as
(select max(SYSTEMDATE) as last_score_dat
from ?score_output_tbl
where SYSTEMDATE < (select max(SYSTEMDATE) from ?score_output_tbl)
)
,
lifestage_seg_cnt_prev AS (
select 
concat(PRED_LABEL, '.',PRED_LIFESTAGE) as lifestage_segment,
avg(MBR_AGE) as avg_member_age_prev,
count(distinct GROUP_LOYALTY_MEMBER_NUMBER) as TOTAL_MEMBER_CNT_prev
from ?score_output_tbl
where SYSTEMDATE = (select last_score_dat from last_score_date)
GROUP BY 1
)
,
lifestage_seg_prob_mean_prev as 
(select
avg(_0_SINGLE_PROB) as _0_SINGLE_PROB_MEAN,
avg(_1_COUPLE_NO_KIDS_PROB) as _1_COUPLE_NO_KIDS_PROB_MEAN,
avg(_2_FAMILY_W_BABY_PROB) as _2_FAMILY_W_BABY_PROB_MEAN,
avg(_3_FAMILY_W_CHILDREN_PROB) as _3_FAMILY_W_CHILDREN_PROB_MEAN,
avg(_4_FAMILY_W_BABY_OTHER_CHILDREN_PROB) as _4_FAMILY_W_BABY_OTHER_CHILDREN_PROB_MEAN,
avg(_5_MULTI_ADULT_FAMILY_PROB) as _5_MULTI_ADULT_FAMILY_PROB_MEAN
from ?score_output_tbl
where SYSTEMDATE = (select last_score_dat from last_score_date)
)
,
lifestage_seg_prob_mean_prev_row as (
select '0.SINGLE' as lifestage_segment, _0_SINGLE_PROB_MEAN as AVG_PROB_prev from lifestage_seg_prob_mean_prev
union
select '1.COUPLE_NO_KIDS' as lifestage_segment, _1_COUPLE_NO_KIDS_PROB_MEAN as AVG_PROB_prev from lifestage_seg_prob_mean_prev
union
select '2.FAMILY_W_BABY' as lifestage_segment, _2_FAMILY_W_BABY_PROB_MEAN as AVG_PROB_prev from lifestage_seg_prob_mean_prev
union
select '3.FAMILY_W_CHILDREN' as lifestage_segment, _3_FAMILY_W_CHILDREN_PROB_MEAN as AVG_PROB_prev from lifestage_seg_prob_mean_prev
union
select '4.FAMILY_W_BABY_OTHER_CHILDREN' as lifestage_segment, _4_FAMILY_W_BABY_OTHER_CHILDREN_PROB_MEAN as AVG_PROB_prev from lifestage_seg_prob_mean_prev
union
select '5.MULTI-ADULT_FAMILY' as lifestage_segment,_5_MULTI_ADULT_FAMILY_PROB_MEAN as AVG_PROB_prev from lifestage_seg_prob_mean_prev
)


select 
a.lifestage_segment,
round(a.avg_member_age_current,0) as avg_member_age_current,
a.TOTAL_MEMBER_CNT_current,
round(b.AVG_PROB_current,4) as AVG_PROB_current,
round(c.avg_member_age_prev,0) as avg_member_age_prev,
c.TOTAL_MEMBER_CNT_prev,
round(d.AVG_PROB_prev,4) as AVG_PROB_prev
from lifestage_seg_cnt_current a
left join lifestage_seg_prob_mean_current_row b
on a.lifestage_segment = b.lifestage_segment
left join lifestage_seg_cnt_prev c
on a.lifestage_segment = c.lifestage_segment
left join lifestage_seg_prob_mean_prev_row d
on a.lifestage_segment = d.lifestage_segment
ORDER BY a.lifestage_segment