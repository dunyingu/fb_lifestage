CREATE OR REPLACE TABLE ?score_mbr_txn_summ_tbl as (
SELECT
a.MBR_NO as GROUP_LOYALTY_MEMBER_NUMBER,
a.SYSTEMDATE,
count(distinct txn.TRANSACTION_IDENTIFIER) AS total_txn_cnt,
sum(txn.TOTAL_SALES_AMOUNT) AS total_txn_sales_amount,
total_txn_sales_amount/total_txn_cnt as avg_txn_sales_amount
from
?src_active_mbr_base a
left join 
(SELECT
GROUP_LOYALTY_MEMBER_NUMBER,
TRANSACTION_DATE, 
TRANSACTION_IDENTIFIER,
TOTAL_SALES_AMOUNT
FROM ?src_txn_tbl 
WHERE GROUP_LOYALTY_MEMBER_NUMBER IS NOT NULL 
) txn
on a.MBR_NO = txn.GROUP_LOYALTY_MEMBER_NUMBER
and txn.TRANSACTION_DATE >= DATEADD('MONTH',-6,a.SYSTEMDATE)
and txn.TRANSACTION_DATE <= DATEADD('DAY',-1,a.SYSTEMDATE)
where a.ACTIVE_12M = 'Y'
group by 1,2
)