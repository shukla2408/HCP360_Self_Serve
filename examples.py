# examples.py
SQL_EXAMPLES = [
    {
        "input": "List the top {top_k} HCPs by total works with their name and NPI.",
        "query": """
        SELECT p.pres_eid, p.first_name, p.last_name, p.npi_number, p.total_works
        FROM dbo.hcp360_persona p
        ORDER BY p.total_works DESC NULLS LAST
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "List HCPs who appear in any scientific study.",
    #     "query": """
    #     SELECT DISTINCT p.pres_eid, p.first_name, p.last_name
    #     FROM dbo.hcp360_persona_scientific_studies s
    #     JOIN dbo.hcp360_persona p ON p.pres_eid = s.pres_eid
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "Show HCPs in <state> with email opt-out enabled.",
        "query": """
        SELECT pres_eid, first_name, last_name, email, email_opt_out_flag
        FROM dbo.hcp360_persona
        WHERE state = <state> AND COALESCE(email_opt_out_flag,'N') = 'Y'
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Count HCPs per therapeutic area (TA).",
    #     "query": """
    #     SELECT s.ta, COUNT(DISTINCT s.pres_eid) AS hcps
    #     FROM dbo.hcp360_persona_segment s
    #     GROUP BY s.ta
    #     ORDER BY hcps DESC
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "Top {top_k} HCPs by number of clinical trials they participated in.",
        "query": """
        SELECT p.pres_eid, p.first_name, p.last_name, p.total_clinical_trials
        FROM dbo.hcp360_persona p
        ORDER BY p.total_clinical_trials DESC NULLS LAST
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Find HCPs who published in <year> and their TA.",
    #     "query": """
    #     SELECT DISTINCT p.pres_eid, p.first_name, p.last_name, seg.ta
    #     FROM dbo.hcp360_persona_scientific_studies s
    #     JOIN dbo.hcp360_persona p ON p.pres_eid = s.pres_eid
    #     LEFT JOIN dbo.hcp360_persona_segment seg ON seg.pres_eid = p.pres_eid
    #     WHERE EXTRACT(YEAR FROM COALESCE(s.publication_date, s.last_update_posted::date)) = <year>
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Average publications per HCP by TA.",
    #     "query": """
    #     WITH pubs AS (
    #       SELECT p.pres_eid, COUNT(*) AS pub_cnt
    #       FROM dbo.hcp360_persona_scientific_studies s
    #       JOIN dbo.hcp360_persona p ON p.pres_eid = s.pres_eid
    #       GROUP BY p.pres_eid
    #     )
    #     SELECT seg.ta, AVG(pubs.pub_cnt)::FLOAT AS avg_pubs_per_hcp
    #     FROM pubs
    #     JOIN dbo.hcp360_persona_segment seg ON seg.pres_eid = pubs.pres_eid
    #     GROUP BY seg.ta
    #     ORDER BY avg_pubs_per_hcp DESC
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Monthly TRx and NRx counts for 'atenolol' in <year>.",
    #     "query": """
    #     SELECT DATE_TRUNC('month', tp_start)::date AS month,
    #            SUM(TRX_COUNT) AS trx, SUM(NRX_COUNT) AS nrx
    #     FROM (
    #       SELECT to_date("TIME_PERIOD_START_DATE",'YYYY-MM-DD') AS tp_start,
    #              "TRX_COUNT", "NRX_COUNT", "PRODUCT_NAME"
    #       FROM dbo.hcp360_prd_rtl_sls
    #     ) x
    #     WHERE x."PRODUCT_NAME" = 'atenolol'
    #       AND EXTRACT(YEAR FROM tp_start) = <year>
    #     GROUP BY 1
    #     ORDER BY 1
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Top territories by TRX_DOLLARS for 'atenolol' between dates.",
    #     "query": """
    #     SELECT "TERRITORY_ID", "TERRITORY_NAME", SUM("TRX_DOLLARS") AS trx_dollars
    #     FROM dbo.hcp360_prd_rtl_sls
    #     WHERE "PRODUCT_NAME" = 'atenolol'
    #       AND to_date("TIME_PERIOD_START_DATE",'YYYY-MM-DD') >= <start_date>
    #       AND to_date("TIME_PERIOD_END_DATE",'YYYY-MM-DD')   <= <end_date>
    #     GROUP BY "TERRITORY_ID", "TERRITORY_NAME"
    #     ORDER BY trx_dollars DESC
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "HCPs with highest NAIVE_MARKET_SHARE in 'Cardiology'.",
    #     "query": """
    #     SELECT s."TA", s."PRES_EID", p.first_name, p.last_name,
    #            AVG(s."NAIVE_MARKET_SHARE") AS avg_naive_share
    #     FROM dbo.hcp360_prd_rtl_sls s
    #     JOIN dbo.hcp360_persona p ON p.pres_eid = s."PRES_EID"
    #     WHERE s."TA" = 'Cardiology'
    #     GROUP BY s."TA", s."PRES_EID", p.first_name, p.last_name
    #     ORDER BY avg_naive_share DESC
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "How many unique HCPs engaged by channel last month?",
        "query": """
        SELECT e."CHANNEL", COUNT(DISTINCT e."PRES_EID") AS unique_hcps
        FROM dbo.hcp360_prsnl_engmnt e
        WHERE e."TRANSACTION_DATETIME" >= date_trunc('month', CURRENT_DATE) - INTERVAL '1 month'
          AND e."TRANSACTION_DATETIME" <  date_trunc('month', CURRENT_DATE)
        GROUP BY e."CHANNEL"
        ORDER BY unique_hcps DESC
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Email engagement funnel (sent, opened, clicked) by TA.",
    #     "query": """
    #     SELECT seg.ta,
    #            SUM(CASE WHEN e."EMAIL_SENT"='Y' THEN 1 ELSE 0 END) AS sent,
    #            SUM(CASE WHEN e."EMAIL_OPENED"='Y' THEN 1 ELSE 0 END) AS opened,
    #            SUM(CASE WHEN e."EMAIL_CLICKED"='Y' THEN 1 ELSE 0 END) AS clicked
    #     FROM dbo.hcp360_prsnl_engmnt e
    #     JOIN dbo.hcp360_persona_segment seg ON seg.pres_eid = e."PRES_EID"
    #     GROUP BY seg.ta
    #     ORDER BY sent DESC
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Top {top_k} products by TRX_DOLLARS growth (current vs prior bucket).",
    #     "query": """
    #     WITH by_bucket AS (
    #       SELECT "PRODUCT_NAME", "TIME_PERIOD_BUCKET",
    #              SUM("TRX_DOLLARS") AS trx_dollars
    #       FROM dbo.hcp360_prd_rtl_sls
    #       GROUP BY 1,2
    #     ),
    #     ranked AS (
    #       SELECT *,
    #              LAG(trx_dollars) OVER (PARTITION BY "PRODUCT_NAME" ORDER BY "TIME_PERIOD_BUCKET") AS prev_dollars
    #       FROM by_bucket
    #     )
    #     SELECT "PRODUCT_NAME",
    #            ("trx_dollars" - COALESCE(prev_dollars,0)) AS delta_dollars
    #     FROM ranked
    #     WHERE prev_dollars IS NOT NULL
    #     ORDER BY delta_dollars DESC
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "Which HCPs had in-person detail calls but never opened emails?",
        "query": """
        SELECT DISTINCT p.pres_eid, p.first_name, p.last_name
        FROM dbo.hcp360_prsnl_engmnt e
        JOIN dbo.hcp360_persona p ON p.pres_eid = e."PRES_EID"
        WHERE e."CHANNEL" = 'In-Person'
          AND NOT EXISTS (
            SELECT 1 FROM dbo.hcp360_prsnl_engmnt e2
            WHERE e2."PRES_EID" = e."PRES_EID" AND e2."EMAIL_OPENED" = 'Y'
          )
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Average TRX per HCP and TA (rollup to overall).",
    #     "query": """
    #     SELECT COALESCE(seg.ta,'ALL') AS ta,
    #            AVG(s."TRX_COUNT")::FLOAT AS avg_trx
    #     FROM dbo.hcp360_prd_rtl_sls s
    #     LEFT JOIN dbo.hcp360_persona_segment seg ON seg.pres_eid = s."PRES_EID"
    #     GROUP BY ROLLUP (seg.ta)
    #     ORDER BY ta
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Distribution of study funding amount by TA (median and p90).",
    #     "query": """
    #     SELECT seg.ta,
    #            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY s.amount) AS median_amount,
    #            PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY s.amount) AS p90_amount
    #     FROM dbo.hcp360_persona_scientific_studies s
    #     JOIN dbo.hcp360_persona_segment seg ON seg.pres_eid = s.pres_eid
    #     WHERE s.amount IS NOT NULL
    #     GROUP BY seg.ta
    #     ORDER BY median_amount DESC
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "Top reps by number of completed calls last quarter.",
        "query": """
        SELECT e."REP_ID", e."REP_NAME", COUNT(*) AS completed_calls
        FROM dbo.hcp360_prsnl_engmnt e
        WHERE e."STATUS_VOD__C" = 'Completed'
          AND e."TRANSACTION_DATETIME" >= date_trunc('quarter', CURRENT_DATE) - INTERVAL '1 quarter'
          AND e."TRANSACTION_DATETIME" <  date_trunc('quarter', CURRENT_DATE)
        GROUP BY e."REP_ID", e."REP_NAME"
        ORDER BY completed_calls DESC
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Which HCPs switched to the product 'atenolol' during <bucket>?",
    #     "query": """
    #     SELECT DISTINCT p.pres_eid, p.first_name, p.last_name
    #     FROM dbo.hcp360_prd_rtl_sls s
    #     JOIN dbo.hcp360_persona p ON p.pres_eid = s."PRES_EID"
    #     WHERE s."PRODUCT_NAME" = 'atenolol'
    #       AND s."TIME_PERIOD_BUCKET" = <bucket>
    #       AND s."SWITCH_TO_PRESCRIPTIONS" > 0
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "Find HCPs with > <min_trials> trials and > <min_publications> publications.",
        "query": """
        SELECT p.pres_eid, p.first_name, p.last_name, p.total_clinical_trials, p.total_works
        FROM dbo.hcp360_persona p
        WHERE COALESCE(p.total_clinical_trials,0) > <min_trials>
          AND COALESCE(p.total_works,0) > <min_publications>
        ORDER BY p.total_clinical_trials DESC
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "For each TA, show top product by TRX_DOLLARS in <bucket>.",
    #     "query": """
    #     SELECT ta, product_name, trx_dollars
    #     FROM (
    #       SELECT s."TA" AS ta, s."PRODUCT_NAME" AS product_name,
    #              SUM(s."TRX_DOLLARS") AS trx_dollars,
    #              ROW_NUMBER() OVER (PARTITION BY s."TA" ORDER BY SUM(s."TRX_DOLLARS") DESC) AS rn
    #       FROM dbo.hcp360_prd_rtl_sls s
    #       WHERE s."TIME_PERIOD_BUCKET" = <bucket>
    #       GROUP BY s."TA", s."PRODUCT_NAME"
    #     ) ranked
    #     WHERE rn = 1
    #     ORDER BY trx_dollars DESC
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "Show campaigns that had click-throughs but no opens (data quality check).",
        "query": """
        SELECT DISTINCT "CAMPAIGN_TYPE"
        FROM dbo.hcp360_prsnl_engmnt
        WHERE "EMAIL_CLICKED" = 'Y' AND COALESCE("EMAIL_OPENED",'N') <> 'Y'
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "HCPs with at least one payment over <amount> tied to 'Atenolol'.",
    #     "query": """
    #     SELECT DISTINCT p.pres_eid, p.first_name, p.last_name, s.associated_drug, s.amount
    #     FROM dbo.hcp360_persona_scientific_studies s
    #     JOIN dbo.hcp360_persona p ON p.pres_eid = s.pres_eid
    #     WHERE s.associated_drug = 'Atenolol' AND s.amount >= <amount>
    #     ORDER BY s.amount DESC
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Which TA has the highest average NBRX_MARKET_SHARE?",
    #     "query": """
    #     SELECT s."TA", AVG(s."NBRX_MARKET_SHARE")::FLOAT AS avg_share
    #     FROM dbo.hcp360_prd_rtl_sls s
    #     GROUP BY s."TA"
    #     ORDER BY avg_share DESC
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Top {top_k} HCPs by TRX_DOLLARS across all products.",
    #     "query": """
    #     SELECT s."PRES_EID", p.first_name, p.last_name, SUM(s."TRX_DOLLARS") AS total_trx_dollars
    #     FROM dbo.hcp360_prd_rtl_sls s
    #     JOIN dbo.hcp360_persona p ON p.pres_eid = s."PRES_EID"
    #     GROUP BY s."PRES_EID", p.first_name, p.last_name
    #     ORDER BY total_trx_dollars DESC
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "Email subject lines with the highest open rate.",
        "query": """
        SELECT "EMAIL_SUBJECT",
               SUM(CASE WHEN "EMAIL_SENT"='Y' THEN 1 ELSE 0 END) AS sent,
               SUM(CASE WHEN "EMAIL_OPENED"='Y' THEN 1 ELSE 0 END) AS opened,
               (SUM(CASE WHEN "EMAIL_OPENED"='Y' THEN 1 ELSE 0 END)::float /
                NULLIF(SUM(CASE WHEN "EMAIL_SENT"='Y' THEN 1 ELSE 0 END),0)) AS open_rate
        FROM dbo.hcp360_prsnl_engmnt
        GROUP BY "EMAIL_SUBJECT"
        HAVING SUM(CASE WHEN "EMAIL_SENT"='Y' THEN 1 ELSE 0 END) >= <min_sends>
        ORDER BY open_rate DESC NULLS LAST
        LIMIT {top_k}
        """
    },
    {
        "input": "HCPs with recent congress attendance but zero emails clicked.",
        "query": """
        SELECT p.pres_eid, p.first_name, p.last_name, p.total_congresses
        FROM dbo.hcp360_persona p
        WHERE COALESCE(p.total_congresses,0) > 0
          AND NOT EXISTS (
            SELECT 1 FROM dbo.hcp360_prsnl_engmnt e
            WHERE e."PRES_EID" = p.pres_eid AND e."EMAIL_CLICKED"='Y'
          )
        ORDER BY p.total_congresses DESC
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Products with the highest NET_SWITCH_VOLUME in <bucket>.",
    #     "query": """
    #     SELECT "PRODUCT_NAME", SUM("NET_SWITCH_VOLUME") AS net_switch
    #     FROM dbo.hcp360_prd_rtl_sls
    #     WHERE "TIME_PERIOD_BUCKET" = <bucket>
    #     GROUP BY "PRODUCT_NAME"
    #     ORDER BY net_switch DESC
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "Trend of email clicks by device type per month in <year>.",
        "query": """
        SELECT DATE_TRUNC('month',"TRANSACTION_DATETIME") AS month,
               "DEVICE_TYPE_VOD__C",
               SUM(CASE WHEN "EMAIL_CLICKED"='Y' THEN 1 ELSE 0 END) AS clicks
        FROM dbo.hcp360_prsnl_engmnt
        WHERE EXTRACT(YEAR FROM "TRANSACTION_DATETIME") = <year>
        GROUP BY 1,2
        ORDER BY 1,2
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Show HCPs who received samples of 'atenolol' but have below-median TRX for it.",
    #     "query": """
    #     WITH trx AS (
    #       SELECT "PRES_EID", SUM("TRX_COUNT") AS trx
    #       FROM dbo.hcp360_prd_rtl_sls
    #       WHERE "PRODUCT_NAME" = 'atenolol'
    #       GROUP BY "PRES_EID"
    #     ), med AS (
    #       SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY trx) AS med_trx FROM trx
    #     )
    #     SELECT p.pres_eid, p.first_name, p.last_name, t.trx
    #     FROM dbo.hcp360_prsnl_engmnt e
    #     JOIN trx t ON t."PRES_EID" = e."PRES_EID"
    #     JOIN dbo.hcp360_persona p ON p.pres_eid = e."PRES_EID"
    #     CROSS JOIN med
    #     WHERE e."SAMPLE_NAME" ILIKE '%' || 'atenolol' || '%'
    #       AND t.trx < med.med_trx
    #     GROUP BY p.pres_eid, p.first_name, p.last_name, t.trx
    #     ORDER BY t.trx ASC
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Find TAs where average email open rate exceeds <threshold>.",
    #     "query": """
    #     WITH by_hcp AS (
    #       SELECT e."PRES_EID",
    #              SUM(CASE WHEN e."EMAIL_SENT"='Y' THEN 1 ELSE 0 END) AS sent,
    #              SUM(CASE WHEN e."EMAIL_OPENED"='Y' THEN 1 ELSE 0 END) AS opened
    #       FROM dbo.hcp360_prsnl_engmnt e
    #       GROUP BY e."PRES_EID"
    #     )
    #     SELECT seg.ta,
    #            AVG((opened::float/NULLIF(sent,0))) AS avg_open_rate
    #     FROM by_hcp h
    #     JOIN dbo.hcp360_persona_segment seg ON seg.pres_eid = h."PRES_EID"
    #     GROUP BY seg.ta
    #     HAVING AVG((opened::float/NULLIF(sent,0))) > <threshold>
    #     ORDER BY avg_open_rate DESC
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Top {top_k} HCPs by biologics sales across all products.",
    #     "query": """
    #     SELECT s."PRES_EID", p.first_name, p.last_name, SUM(s."BIOLOGICS_SALES") AS bio_sales
    #     FROM dbo.hcp360_prd_rtl_sls s
    #     JOIN dbo.hcp360_persona p ON p.pres_eid = s."PRES_EID"
    #     GROUP BY s."PRES_EID", p.first_name, p.last_name
    #     ORDER BY bio_sales DESC
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "Which channels drive the most clicks for 'Awareness'?",
        "query": """
        SELECT e."CHANNEL",
               SUM(CASE WHEN e."EMAIL_CLICKED"='Y' THEN 1 ELSE 0 END) AS clicks
        FROM dbo.hcp360_prsnl_engmnt e
        WHERE e."CAMPAIGN_TYPE" = 'Awareness'
        GROUP BY e."CHANNEL"
        ORDER BY clicks DESC
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Cross-sell: HCPs prescribing <product_a> and <product_b>.",
    #     "query": """
    #     SELECT p.pres_eid, p.first_name, p.last_name
    #     FROM dbo.hcp360_persona p
    #     WHERE EXISTS (
    #       SELECT 1 FROM dbo.hcp360_prd_rtl_sls s
    #       WHERE s."PRES_EID" = p.pres_eid AND s."PRODUCT_NAME" = <product_a>
    #     ) AND EXISTS (
    #       SELECT 1 FROM dbo.hcp360_prd_rtl_sls s2
    #       WHERE s2."PRES_EID" = p.pres_eid AND s2."PRODUCT_NAME" = <product_b>
    #     )
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Year-over-year TRX growth by product in 'Cardiology'.",
    #     "query": """
    #     WITH by_year AS (
    #       SELECT "PRODUCT_NAME", "TA",
    #              EXTRACT(YEAR FROM to_date("TIME_PERIOD_START_DATE",'YYYY-MM-DD'))::int AS yr,
    #              SUM("TRX_COUNT") AS trx
    #       FROM dbo.hcp360_prd_rtl_sls
    #       GROUP BY 1,2,3
    #     )
    #     SELECT a."PRODUCT_NAME", a.yr AS year, a.trx, b.trx AS prev_trx,
    #            (a.trx - b.trx) AS yoy_delta
    #     FROM by_year a
    #     LEFT JOIN by_year b
    #       ON b."PRODUCT_NAME" = a."PRODUCT_NAME"
    #      AND b."TA" = a."TA"
    #      AND b.yr = a.yr - 1
    #     WHERE a."TA" = 'Cardiology'
    #     ORDER BY yoy_delta DESC NULLS LAST
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "HCPs with no segment assigned (data gap).",
    #     "query": """
    #     SELECT p.pres_eid, p.first_name, p.last_name
    #     FROM dbo.hcp360_persona p
    #     LEFT JOIN dbo.hcp360_persona_segment s ON s.pres_eid = p.pres_eid
    #     WHERE s.pres_eid IS NULL
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Most referenced associated_drug in studies per TA.",
    #     "query": """
    #     SELECT seg.ta, s.associated_drug, COUNT(*) AS cnt
    #     FROM dbo.hcp360_persona_scientific_studies s
    #     JOIN dbo.hcp360_persona_segment seg ON seg.pres_eid = s.pres_eid
    #     GROUP BY seg.ta, s.associated_drug
    #     ORDER BY cnt DESC
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "Find emails sent without valid consent.",
        "query": """
        SELECT *
        FROM dbo.hcp360_prsnl_engmnt
        WHERE "CHANNEL" = 'Email' AND COALESCE("VALID_CONSENT_EXIST",'N') <> 'Y'
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Correlate samples and TRX: avg TRX by SAMPLE_COUNT bucket.",
    #     "query": """
    #     WITH s AS (
    #       SELECT "PRES_EID",
    #              CASE
    #                WHEN "SAMPLE_COUNT" >= 20 THEN '20+'
    #                WHEN "SAMPLE_COUNT" >= 10 THEN '10-19'
    #                WHEN "SAMPLE_COUNT" >= 5  THEN '5-9'
    #                WHEN "SAMPLE_COUNT" >= 1  THEN '1-4'
    #                ELSE '0'
    #              END AS sample_bucket
    #       FROM dbo.hcp360_prsnl_engmnt
    #     ), t AS (
    #       SELECT "PRES_EID", SUM("TRX_COUNT") AS trx
    #       FROM dbo.hcp360_prd_rtl_sls
    #       GROUP BY "PRES_EID"
    #     )
    #     SELECT s.sample_bucket, AVG(t.trx)::float AS avg_trx
    #     FROM s
    #     JOIN t ON t."PRES_EID" = s."PRES_EID"
    #     GROUP BY s.sample_bucket
    #     ORDER BY sample_bucket
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Top {top_k} territories by combined TRX and NRX units.",
    #     "query": """
    #     SELECT "TERRITORY_ID", "TERRITORY_NAME",
    #            SUM("TRX_UNIT" + "NRX_UNIT") AS total_units
    #     FROM dbo.hcp360_prd_rtl_sls
    #     GROUP BY "TERRITORY_ID", "TERRITORY_NAME"
    #     ORDER BY total_units DESC
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "HCPs who clicked at least one link in the last 30 days.",
        "query": """
        SELECT DISTINCT e."PRES_EID", p.first_name, p.last_name
        FROM dbo.hcp360_prsnl_engmnt e
        JOIN dbo.hcp360_persona p ON p.pres_eid = e."PRES_EID"
        WHERE e."EMAIL_CLICKED"='Y'
          AND e."EMAIL_LAST_CLICKED_DATE" >= CURRENT_DATE - INTERVAL '30 days'
        LIMIT {top_k}
        """
    },
    {
        "input": "Compare open rate by device for 'atenolol'.",
        "query": """
        SELECT e."DEVICE_TYPE_VOD__C",
               AVG(CASE WHEN e."EMAIL_OPENED"='Y' THEN 1 ELSE 0 END)::float AS open_rate
        FROM dbo.hcp360_prsnl_engmnt e
        WHERE e."PRODUCT_NAME" = 'atenolol'
        GROUP BY e."DEVICE_TYPE_VOD__C"
        ORDER BY open_rate DESC NULLS LAST
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Longest running studies per TA (by duration days).",
    #     "query": """
    #     SELECT seg.ta, s.study_title,
    #            (COALESCE(s.primary_completion_date, s.end_date)::date
    #            - COALESCE(s.start_date, s.last_update_submitted)::date) AS duration_days
    #     FROM dbo.hcp360_persona_scientific_studies s
    #     JOIN dbo.hcp360_persona_segment seg ON seg.pres_eid = s.pres_eid
    #     WHERE s.start_date IS NOT NULL
    #     ORDER BY duration_days DESC NULLS LAST
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Products with highest NBRX to TRX conversion ratio.",
    #     "query": """
    #     SELECT "PRODUCT_NAME",
    #            (SUM("NBRX_COUNT")/NULLIF(SUM("TRX_COUNT"),0))::float AS nbrx_trx_ratio
    #     FROM dbo.hcp360_prd_rtl_sls
    #     GROUP BY "PRODUCT_NAME"
    #     HAVING SUM("TRX_COUNT") > 0
    #     ORDER BY nbrx_trx_ratio DESC
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "Time spent distribution by call type.",
        "query": """
        SELECT "CALL_TYPE",
               PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY (REGEXP_REPLACE("TIME_SPENT",'\\D','','g')::int)) AS p50_mins,
               PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY (REGEXP_REPLACE("TIME_SPENT",'\\D','','g')::int)) AS p90_mins
        FROM dbo.hcp360_prsnl_engmnt
        GROUP BY "CALL_TYPE"
        ORDER BY p50_mins DESC NULLS LAST
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Top HCPs by NAIVE_VOLUME_GROWTH for 'atenolol'.",
    #     "query": """
    #     SELECT "PRES_EID", SUM("NAIVE_VOLUME_GROWTH") AS growth
    #     FROM dbo.hcp360_prd_rtl_sls
    #     WHERE "PRODUCT_NAME" = 'atenolol'
    #     GROUP BY "PRES_EID"
    #     ORDER BY growth DESC
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "HCPs with studies on 'Atenolol' and received 'Atenolol' samples.",
    #     "query": """
    #     SELECT DISTINCT p.pres_eid, p.first_name, p.last_name
    #     FROM dbo.hcp360_persona_scientific_studies s
    #     JOIN dbo.hcp360_persona p ON p.pres_eid = s.pres_eid
    #     JOIN dbo.hcp360_prsnl_engmnt e ON e."PRES_EID" = p.pres_eid
    #     WHERE s.associated_drug = 'Atenolol'
    #       AND e."SAMPLE_NAME" ILIKE '%' || 'Atenolol' || '%'
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "Top {top_k} HCPs by combined email clicks and in-person calls.",
        "query": """
        SELECT e."PRES_EID", p.first_name, p.last_name,
               SUM(CASE WHEN e."EMAIL_CLICKED"='Y' THEN 1 ELSE 0 END)
               + SUM(CASE WHEN e."CHANNEL"='In-Person' THEN 1 ELSE 0 END) AS engagement_score
        FROM dbo.hcp360_prsnl_engmnt e
        JOIN dbo.hcp360_persona p ON p.pres_eid = e."PRES_EID"
        GROUP BY e."PRES_EID", p.first_name, p.last_name
        ORDER BY engagement_score DESC
        LIMIT {top_k}
        """
    },
    {
        "input": "Days since last engagement by HCP.",
        "query": """
        SELECT e."PRES_EID", p.first_name, p.last_name,
               (CURRENT_DATE - MAX(e."TRANSACTION_DATETIME")) AS days_since_last
        FROM dbo.hcp360_prsnl_engmnt e
        JOIN dbo.hcp360_persona p ON p.pres_eid = e."PRES_EID"
        GROUP BY e."PRES_EID", p.first_name, p.last_name
        ORDER BY days_since_last DESC NULLS LAST
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Show HCPs with high TRX but low market share for 'atenolol'.",
    #     "query": """
    #     WITH t AS (
    #       SELECT "PRES_EID", SUM("TRX_COUNT") AS trx, AVG("TRX_BOTTLE_MARKETSHARE") AS mkt_share
    #       FROM dbo.hcp360_prd_rtl_sls
    #       WHERE "PRODUCT_NAME" = 'atenolol'
    #       GROUP BY "PRES_EID"
    #     ), med AS (
    #       SELECT PERCENTILE_CONT(0.7) WITHIN GROUP (ORDER BY trx) AS p70_trx FROM t
    #     )
    #     SELECT p.pres_eid, p.first_name, p.last_name, t.trx, t.mkt_share
    #     FROM t
    #     JOIN med ON TRUE
    #     JOIN dbo.hcp360_persona p ON p.pres_eid = t."PRES_EID"
    #     WHERE t.trx >= med.p70_trx AND t.mkt_share < 10
    #     ORDER BY t.trx DESC
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "Top assets by clicks in emails.",
        "query": """
        SELECT "gilead_asset_id",
               SUM(CASE WHEN "EMAIL_CLICKED"='Y' THEN 1 ELSE 0 END) AS clicks
        FROM dbo.hcp360_prsnl_engmnt
        GROUP BY "gilead_asset_id"
        ORDER BY clicks DESC
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Products with highest TRX to dollar ratio (efficiency).",
    #     "query": """
    #     SELECT "PRODUCT_NAME",
    #            (SUM("TRX_COUNT")/NULLIF(SUM("TRX_DOLLARS"),0))::float AS trx_per_dollar
    #     FROM dbo.hcp360_prd_rtl_sls
    #     GROUP BY "PRODUCT_NAME"
    #     HAVING SUM("TRX_DOLLARS") > 0
    #     ORDER BY trx_per_dollar DESC
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Which TA shows the largest increase in email opens month-over-month?",
    #     "query": """
    #     WITH by_month AS (
    #       SELECT seg.ta, DATE_TRUNC('month', e."EMAIL_LAST_OPEN_DATE") AS mth,
    #              COUNT(*) FILTER (WHERE e."EMAIL_OPENED"='Y') AS opens
    #       FROM dbo.hcp360_prsnl_engmnt e
    #       JOIN dbo.hcp360_persona_segment seg ON seg.pres_eid = e."PRES_EID"
    #       GROUP BY seg.ta, DATE_TRUNC('month', e."EMAIL_LAST_OPEN_DATE")
    #     ), ranked AS (
    #       SELECT *, LAG(opens) OVER (PARTITION BY ta ORDER BY mth) AS prev_opens
    #       FROM by_month
    #     )
    #     SELECT ta, mth::date AS month, (opens - COALESCE(prev_opens,0)) AS delta_opens
    #     FROM ranked
    #     WHERE prev_opens IS NOT NULL
    #     ORDER BY delta_opens DESC
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "Rep productivity: completed vs planned calls per rep.",
        "query": """
        SELECT "REP_ID", "REP_NAME",
               COUNT(*) FILTER (WHERE "STATUS_VOD__C"='Completed') AS completed,
               COUNT(*) AS total
        FROM dbo.hcp360_prsnl_engmnt
        GROUP BY "REP_ID", "REP_NAME"
        ORDER BY completed DESC
        LIMIT {top_k}
        """
    },
    {
        "input": "Email open latency (days) distribution for 'Awareness'.",
        "query": """
        SELECT PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM ("EMAIL_LAST_OPEN_DATE"::timestamp - "TRANSACTION_DATETIME"))/86400) AS p50_days,
               PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY EXTRACT(EPOCH FROM ("EMAIL_LAST_OPEN_DATE"::timestamp - "TRANSACTION_DATETIME"))/86400) AS p90_days
        FROM dbo.hcp360_prsnl_engmnt
        WHERE "CAMPAIGN_TYPE" ='Awareness'
          AND "EMAIL_OPENED"='Y'
        LIMIT {top_k}
        """
    },
    {
        "input": "HCPs with no engagements in the last <n_days> days.",
        "query": """
        SELECT p.pres_eid, p.first_name, p.last_name
        FROM dbo.hcp360_persona p
        WHERE NOT EXISTS (
          SELECT 1 FROM dbo.hcp360_prsnl_engmnt e
          WHERE e."PRES_EID" = p.pres_eid
            AND e."TRANSACTION_DATETIME" >= CURRENT_DATE - (<n_days> || ' days')::interval
        )
        LIMIT {top_k}
        """
    },
    {
        "input": "Show documents clicked (URL) for Atenolol in the last month.",
        "query": """
        SELECT DISTINCT e."CLICK_URL_VOD__C"
        FROM dbo.hcp360_prsnl_engmnt e
        WHERE e."PRODUCT_NAME" = 'Atenolol'
          AND e."EMAIL_CLICKED"='Y'
          AND e."EMAIL_LAST_CLICKED_DATE" >= date_trunc('month', CURRENT_DATE) - INTERVAL '1 month'
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Segment-level mix: share of HCPs by channel preference (most used channel).",
    #     "query": """
    #     WITH ch AS (
    #       SELECT "PRES_EID", "CHANNEL", COUNT(*) AS cnt,
    #              ROW_NUMBER() OVER (PARTITION BY "PRES_EID" ORDER BY COUNT(*) DESC) AS rn
    #       FROM dbo.hcp360_prsnl_engmnt
    #       GROUP BY "PRES_EID","CHANNEL"
    #     )
    #     SELECT seg.ta, ch."CHANNEL", COUNT(*) AS hcps
    #     FROM ch
    #     JOIN dbo.hcp360_persona_segment seg ON seg.pres_eid = ch."PRES_EID"
    #     WHERE ch.rn = 1
    #     GROUP BY seg.ta, ch."CHANNEL"
    #     ORDER BY hcps DESC
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Top {top_k} study sponsors by total amount paid.",
    #     "query": """
    #     SELECT s.payer_company, SUM(s.amount) AS total_amount
    #     FROM dbo.hcp360_persona_scientific_studies s
    #     GROUP BY s.payer_company
    #     ORDER BY total_amount DESC NULLS LAST
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "Email performance by territory for Atenolol.",
        "query": """
        SELECT e."TERRITORY_ID", e."TERRITORY_NAME",
               SUM(CASE WHEN e."EMAIL_SENT"='Y' THEN 1 ELSE 0 END) AS sent,
               SUM(CASE WHEN e."EMAIL_OPENED"='Y' THEN 1 ELSE 0 END) AS opened,
               SUM(CASE WHEN e."EMAIL_CLICKED"='Y' THEN 1 ELSE 0 END) AS clicked
        FROM dbo.hcp360_prsnl_engmnt e
        WHERE e."PRODUCT_NAME" = 'atenolol'
        GROUP BY e."TERRITORY_ID", e."TERRITORY_NAME"
        ORDER BY clicked DESC
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Which HCPs increased NAÏVE_VOLUME quarter-over-quarter?",
    #     "query": """
    #     WITH q AS (
    #       SELECT "PRES_EID",
    #              "TIME_PERIOD_BUCKET",
    #              SUM("NAÏVE_VOLUME") AS naive_vol
    #       FROM dbo.hcp360_prd_rtl_sls
    #       GROUP BY 1,2
    #     ), r AS (
    #       SELECT *, LAG(naive_vol) OVER (PARTITION BY "PRES_EID" ORDER BY "TIME_PERIOD_BUCKET") AS prev_vol
    #       FROM q
    #     )
    #     SELECT "PRES_EID", "TIME_PERIOD_BUCKET", naive_vol, prev_vol, (naive_vol - prev_vol) AS delta
    #     FROM r
    #     WHERE prev_vol IS NOT NULL AND naive_vol > prev_vol
    #     ORDER BY delta DESC
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Top {top_k} HCPs by combined retail sales and engagement clicks.",
    #     "query": """
    #     WITH sales AS (
    #       SELECT "PRES_EID", SUM("TRX_DOLLARS") AS dollars
    #       FROM dbo.hcp360_prd_rtl_sls
    #       GROUP BY "PRES_EID"
    #     ), clicks AS (
    #       SELECT "PRES_EID", SUM(CASE WHEN "EMAIL_CLICKED"='Y' THEN 1 ELSE 0 END) AS clicks
    #       FROM dbo.hcp360_prsnl_engmnt
    #       GROUP BY "PRES_EID"
    #     )
    #     SELECT p.pres_eid, p.first_name, p.last_name,
    #            COALESCE(sales.dollars,0) AS dollars,
    #            COALESCE(clicks.clicks,0) AS clicks,
    #            (COALESCE(sales.dollars,0) * 0.001 + COALESCE(clicks.clicks,0)) AS score
    #     FROM dbo.hcp360_persona p
    #     LEFT JOIN sales ON sales."PRES_EID" = p.pres_eid
    #     LEFT JOIN clicks ON clicks."PRES_EID" = p.pres_eid
    #     ORDER BY score DESC
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "Identify HCPs with conflicting consent and email activity (clicked but no consent).",
        "query": """
        SELECT DISTINCT e."PRES_EID"
        FROM dbo.hcp360_prsnl_engmnt e
        WHERE e."EMAIL_CLICKED"='Y' AND COALESCE(e."VALID_CONSENT_EXIST",'N') <> 'Y'
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Top {top_k} products by NBRX_BOTTLE_MARKETSHARE in 'Cardiology'.",
    #     "query": """
    #     SELECT "PRODUCT_NAME", AVG("NRX_BOTTLE_MARKETSHARE") AS avg_share
    #     FROM dbo.hcp360_prd_rtl_sls
    #     WHERE "TA" = 'Cardiology'
    #     GROUP BY "PRODUCT_NAME"
    #     ORDER BY avg_share DESC
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "HCPs with studies in <year> and received payments that year.",
    #     "query": """
    #     SELECT DISTINCT p.pres_eid, p.first_name, p.last_name
    #     FROM dbo.hcp360_persona_scientific_studies s
    #     JOIN dbo.hcp360_persona p ON p.pres_eid = s.pres_eid
    #     WHERE EXTRACT(YEAR FROM COALESCE(s.payment_date, s.last_update_posted::date)) = <year>
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Engagements and sales joined: TRX per engagement for 'atenolol'.",
    #     "query": """
    #     WITH e AS (
    #       SELECT "PRES_EID", COUNT(*) AS engagements
    #       FROM dbo.hcp360_prsnl_engmnt
    #       WHERE "PRODUCT_NAME" = 'atenolol'
    #       GROUP BY "PRES_EID"
    #     ), s AS (
    #       SELECT "PRES_EID", SUM("TRX_COUNT") AS trx
    #       FROM dbo.hcp360_prd_rtl_sls
    #       WHERE "PRODUCT_NAME" = 'atenolol'
    #       GROUP BY "PRES_EID"
    #     )
    #     SELECT p.pres_eid, p.first_name, p.last_name, s.trx, e.engagements,
    #            (s.trx::float / NULLIF(e.engagements,0)) AS trx_per_engagement
    #     FROM dbo.hcp360_persona p
    #     LEFT JOIN s ON s."PRES_EID" = p.pres_eid
    #     LEFT JOIN e ON e."PRES_EID" = p.pres_eid
    #     ORDER BY trx_per_engagement DESC NULLS LAST
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Find HCPs where territory in sales differs from engagement logs (consistency check).",
    #     "query": """
    #     SELECT DISTINCT s."PRES_EID"
    #     FROM dbo.hcp360_prd_rtl_sls s
    #     JOIN dbo.hcp360_prsnl_engmnt e ON e."PRES_EID" = s."PRES_EID"
    #     WHERE s."TERRITORY_ID" <> e."TERRITORY_ID"
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "Most common indications mentioned in engagements.",
        "query": """
        SELECT "INDICATION", COUNT(*) AS cnt
        FROM dbo.hcp360_prsnl_engmnt
        GROUP BY "INDICATION"
        ORDER BY cnt DESC
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Products with rising email opens but falling TRX (early warning).",
    #     "query": """
    #     WITH opens AS (
    #       SELECT "PRODUCT_NAME",
    #              DATE_TRUNC('month',"EMAIL_LAST_OPEN_DATE") AS mth,
    #              COUNT(*) FILTER (WHERE "EMAIL_OPENED"='Y') AS opened
    #       FROM dbo.hcp360_prsnl_engmnt
    #       GROUP BY 1,2
    #     ), trx AS (
    #       SELECT "PRODUCT_NAME",
    #              DATE_TRUNC('month', to_date("TIME_PERIOD_START_DATE",'YYYY-MM-DD')) AS mth,
    #              SUM("TRX_COUNT") AS trx_cnt
    #       FROM dbo.hcp360_prd_rtl_sls
    #       GROUP BY 1,2
    #     ), chg AS (
    #       SELECT o."PRODUCT_NAME", o.mth,
    #              o.opened - LAG(o.opened) OVER (PARTITION BY o."PRODUCT_NAME" ORDER BY o.mth) AS d_open,
    #              t.trx_cnt - LAG(t.trx_cnt) OVER (PARTITION BY t."PRODUCT_NAME" ORDER BY t.mth) AS d_trx
    #       FROM opens o
    #       JOIN trx t ON t."PRODUCT_NAME" = o."PRODUCT_NAME" AND t.mth = o.mth
    #     )
    #     SELECT "PRODUCT_NAME", mth::date, d_open, d_trx
    #     FROM chg
    #     WHERE d_open > 0 AND d_trx < 0
    #     ORDER BY mth DESC
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "HCPs in 'Cardiology' with above-average TRX_DOLLARS and below-average open rate.",
    #     "query": """
    #     WITH t AS (
    #       SELECT "PRES_EID", AVG("TRX_DOLLARS") AS avg_dollars
    #       FROM dbo.hcp360_prd_rtl_sls
    #       WHERE "TA" = 'Cardiology'
    #       GROUP BY "PRES_EID"
    #     ), o AS (
    #       SELECT "PRES_EID",
    #              AVG(CASE WHEN "EMAIL_OPENED"='Y' THEN 1 ELSE 0 END)::float AS open_rate
    #       FROM dbo.hcp360_prsnl_engmnt
    #       GROUP BY "PRES_EID"
    #     ), agg AS (
    #       SELECT AVG(avg_dollars) AS ta_avg_dollars FROM t
    #     )
    #     SELECT p.pres_eid, p.first_name, p.last_name, t.avg_dollars, o.open_rate
    #     FROM t
    #     JOIN o ON o."PRES_EID" = t."PRES_EID"
    #     JOIN agg ON TRUE
    #     JOIN dbo.hcp360_persona p ON p.pres_eid = t."PRES_EID"
    #     WHERE t.avg_dollars > agg.ta_avg_dollars AND o.open_rate < <open_threshold>
    #     ORDER BY t.avg_dollars DESC
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Study count by study_type and TA with totals (GROUPING SETS).",
    #     "query": """
    #     SELECT COALESCE(seg.ta,'ALL') AS ta, COALESCE(s.study_type,'ALL') AS study_type,
    #            COUNT(*) AS studies
    #     FROM dbo.hcp360_persona_scientific_studies s
    #     LEFT JOIN dbo.hcp360_persona_segment seg ON seg.pres_eid = s.pres_eid
    #     GROUP BY GROUPING SETS ((seg.ta, s.study_type), (seg.ta), ())
    #     ORDER BY ta, study_type
    #     LIMIT {top_k}
    #     """
    # },
    {
        "input": "HCPs with the most distinct products engaged in.",
        "query": """
        SELECT e."PRES_EID", p.first_name, p.last_name, COUNT(DISTINCT e."PRODUCT_NAME") AS products
        FROM dbo.hcp360_prsnl_engmnt e
        JOIN dbo.hcp360_persona p ON p.pres_eid = e."PRES_EID"
        GROUP BY e."PRES_EID", p.first_name, p.last_name
        ORDER BY products DESC
        LIMIT {top_k}
        """
    },
    # {
    #     "input": "Outliers: HCPs with TRX_DOLLARS above p95 in <bucket>.",
    #     "query": """
    #     WITH a AS (
    #       SELECT "PRES_EID", SUM("TRX_DOLLARS") AS dollars
    #       FROM dbo.hcp360_prd_rtl_sls
    #       WHERE "TIME_PERIOD_BUCKET" = <bucket>
    #       GROUP BY "PRES_EID"
    #     ), p AS (
    #       SELECT PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY dollars) AS p95 FROM a
    #     )
    #     SELECT a."PRES_EID", p95, a.dollars
    #     FROM a CROSS JOIN p
    #     WHERE a.dollars >= p.p95
    #     ORDER BY a.dollars DESC
    #     LIMIT {top_k}
    #     """
    # },
    # {
    #     "input": "Top {top_k} cities by study participation (unique HCPs).",
    #     "query": """
    #     SELECT p.city, COUNT(DISTINCT s.pres_eid) AS hcps
    #     FROM dbo.hcp360_persona_scientific_studies s
    #     JOIN dbo.hcp360_persona p ON p.pres_eid = s.pres_eid
    #     GROUP BY p.city
    #     ORDER BY hcps DESC NULLS LAST
    #     LIMIT {top_k}
    #     """
    # }
]
