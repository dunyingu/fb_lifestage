select 
distinct 
to_varchar(a.MBR_NO) as GROUP_LOYALTY_MEMBER_NUMBER,
a.SYSTEMDATE,
case when upper(a.MBR_GENDER)='F' then 0
     when upper(a.MBR_GENDER)='M' then 1 
	 else null end as PRMY_MEMBER_GENDER,
case when a.MBR_AGE<=0 then null
     else a.MBR_AGE end as MEMBER_AGE_AT_SURVEY,
CAST ((CASE WHEN a.CS_RFM_SEGMENT='6.Incidental shoppers' THEN 0
               WHEN a.CS_RFM_SEGMENT='5.Infrequent small shopper' THEN 1
               WHEN a.CS_RFM_SEGMENT='4.Frequent small baskets' THEN 2
               WHEN a.CS_RFM_SEGMENT='3.Occasional bigger spenders' THEN 3
               WHEN a.CS_RFM_SEGMENT='2.Heavy users' THEN 4
               WHEN a.CS_RFM_SEGMENT='1.Mega shoppers' THEN 5
               ELSE NULL END) AS INTEGER) as RFM_SEGMENT,
CAST((case when a.CS_RFM_TILE='1,1' then 1
     when a.CS_RFM_TILE='1,2' then 2
     when a.CS_RFM_TILE='1,3' then 3
     when a.CS_RFM_TILE='1,4' then 4
     when a.CS_RFM_TILE='2,1' then 5
     when a.CS_RFM_TILE='2,2' then 6
     when a.CS_RFM_TILE='2,3' then 7
     when a.CS_RFM_TILE='2,4' then 8
     when a.CS_RFM_TILE='3,1' then 9
     when a.CS_RFM_TILE='3,2' then 10
     when a.CS_RFM_TILE='3,3' then 11
     when a.CS_RFM_TILE='3,4' then 12
     when a.CS_RFM_TILE='4,1' then 13
     when a.CS_RFM_TILE='4,2' then 14
     when a.CS_RFM_TILE='4,3' then 15
     when a.CS_RFM_TILE='4,4' then 16
     else null END) AS integer) as RFM_TILE,
count(DISTINCT b.CARD_NO) AS active_cards
--,c.MOSAIC_LEVEL
--,c.MOSAIC_INCOME_LEVEL
from ?src_active_mbr_base a
left join ?src_active_cards_tbl b
ON a.mbr_no = b.mbr_no
AND a.SYSTEMDATE between b.EFF_FROM_DT and b.EFF_To_DT
--LEFT JOIN (SELECT 
--DISTINCT 
--MBR_NO,
--SNAPSHOT_DATE,
--CAST (RIGHT(MOSAIC,2) AS int)-1 MOSAIC_LEVEL,
--CAST (MOSAIC_INCOME_BAND AS INT)-1 mosaic_income_level
--FROM EXP_ALL.ANALYTICS.ACTIVE_12M_MBR_FULL_BACKUP
--WHERE MOSAIC IS NOT NULL AND MOSAIC NOT LIKE '-%'
--AND MOSAIC_INCOME_BAND IS NOT NULL AND MOSAIC_INCOME_BAND != '99' 
-- and SNAPSHOT_DATE = date('2021-08-30')
--) c
--ON a.mbr_no = c.mbr_no
--and year(a.SYSTEMDATE) = year(c.SNAPSHOT_DATE)
--and quarter(a.SYSTEMDATE) = quarter(c.SNAPSHOT_DATE)
WHERE a.ACTIVE_12M = 'Y'
--and a.MOSAIC IS NOT NULL AND a.MOSAIC NOT LIKE '-%'
--AND a.MOSAIC_INCOME_BAND IS NOT NULL AND a.MOSAIC_INCOME_BAND != '99' 
GROUP BY 1,2,3,4,5,6