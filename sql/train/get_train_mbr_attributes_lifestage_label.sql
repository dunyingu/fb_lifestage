select 
distinct 
a.GROUP_LOYALTY_MEMBER_NUMBER,
a.RESPONSE_DATE,
case when a.PRMY_MEMBER_GENDER='F' then 0
     when a.PRMY_MEMBER_GENDER='M' then 1 
	 else null end as PRMY_MEMBER_GENDER,
case when (a.MEMBER_AGE_AT_SURVEY<=0 or a.MEMBER_AGE_AT_SURVEY>=99) then null
     else a.MEMBER_AGE_AT_SURVEY end as MEMBER_AGE_AT_SURVEY,
CAST ((CASE WHEN a.RFM_SEGMENT='6.Incidental shoppers' THEN 0
               WHEN a.RFM_SEGMENT='5.Infrequent small shopper' THEN 1
               WHEN a.RFM_SEGMENT='4.Frequent small baskets' THEN 2
               WHEN a.RFM_SEGMENT='3.Occasional bigger spenders' THEN 3
               WHEN a.RFM_SEGMENT='2.Heavy users' THEN 4
               WHEN a.RFM_SEGMENT='1.Mega shoppers' THEN 5
               ELSE NULL END) AS INTEGER) as RFM_SEGMENT,
CAST((case when a.RFM_TILE='1,1' then 1
     when a.RFM_TILE='1,2' then 2
     when a.RFM_TILE='1,3' then 3
     when a.RFM_TILE='1,4' then 4
     when a.RFM_TILE='2,1' then 5
     when a.RFM_TILE='2,2' then 6
     when a.RFM_TILE='2,3' then 7
     when a.RFM_TILE='2,4' then 8
     when a.RFM_TILE='3,1' then 9
     when a.RFM_TILE='3,2' then 10
     when a.RFM_TILE='3,3' then 11
     when a.RFM_TILE='3,4' then 12
     when a.RFM_TILE='4,1' then 13
     when a.RFM_TILE='4,2' then 14
     when a.RFM_TILE='4,3' then 15
     when a.RFM_TILE='4,4' then 16
     else null END) AS integer) as RFM_TILE,
count(DISTINCT b.CARD_NO) AS active_cards,
--c.MOSAIC_LEVEL,
--c.mosaic_income_level,
CAST((case when a.LIFESTAGE_LABEL='SINGLE' then 0
     when a.LIFESTAGE_LABEL='COUPLE_NO_KIDS' then 1
     when a.LIFESTAGE_LABEL='FAMILY_W_BABY' then 2
     when a.LIFESTAGE_LABEL='FAMILY_W_CHILDREN' then 3
     when a.LIFESTAGE_LABEL='FAMILY_W_BABY_OTHER_CHILDREN' then 4
     when a.LIFESTAGE_LABEL='MULTI-ADULT_FAMILY' then 5 END) AS integer) as LIFESTAGE_LABEL
from ?train_mbr_agg_diff_txn_tbl a
left join ?src_active_cards_tbl b
ON a.GROUP_LOYALTY_MEMBER_NUMBER = b.mbr_no
AND a.RESPONSE_DATE between b.EFF_FROM_DT and b.EFF_To_DT
--LEFT JOIN (SELECT 
--DISTINCT 
--MBR_NO,
--RESPONSE_DATE
--CAST (substring(mosaic,2,3) AS int)-1 MOSAIC_LEVEL,
--CAST (MOSAIC_INCOME_BAND AS INT)-1 mosaic_income_level
--FROM EXP_ALL.DATA_SCIENCE.GRP_LYLTY_MBR_SRVY_LIFESTAGE_HIST_MOSAIC_DESC_INCOME
--WHERE MOSAIC IS NOT NULL AND MOSAIC NOT LIKE '-%'
-- AND MOSAIC_INCOME_BAND IS NOT NULL AND MOSAIC_INCOME_BAND != '99' 
--) c
--ON a.GROUP_LOYALTY_MEMBER_NUMBER = c.GROUP_LOYALTY_MEMBER_NUMBER
--and a.RESPONSE_DATE = c.RESPONSE_DATE
GROUP BY 1,2,3,4,5,6,8