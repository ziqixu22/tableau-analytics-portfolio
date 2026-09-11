# Tableau Analytics Portfolio

A curated collection of Tableau and business-intelligence projects spanning healthcare, customer analytics, media, and higher-education enrollment data.

This repository is a **portfolio landing page**. The detailed technical methodology, equations, data logic, evaluation strategy, and limitations live in the individual project repositories linked below.

## Portfolio Goal

The purpose of this portfolio is to demonstrate a complete analytics workflow:

```text
Business / research question
        ↓
Define the analytical population
        ↓
Clean / reshape / engineer variables
        ↓
Choose metrics appropriate to the question
        ↓
Build Tableau calculations and views
        ↓
Validate KPI consistency
        ↓
Communicate interpretable results
```

The projects intentionally cover different analytical settings. Not every problem requires machine learning: some are best solved through cohort construction, aggregation, segmentation, or longitudinal visualization.

## Featured Projects

### 1. Flu Shot Analytics Dashboard

**Domain:** Healthcare Analytics  
**Tools:** SQL + Tableau

**Problem:** Measure influenza-vaccination uptake among a consistently defined active-patient population and compare coverage across demographic and geographic groups.

The project builds a patient-level binary outcome

$$
Y_i=
\begin{cases}
1,&\text{patient }i\text{ received a qualifying 2022 flu shot},\\
0,&\text{otherwise},
\end{cases}
$$

and computes coverage as

$$
\widehat p=\frac{1}{N}\sum_{i=1}^{N}Y_i.
$$

A SQL `LEFT JOIN` is crucial because unvaccinated patients must remain in the denominator.

**What it demonstrates:** cohort definition, CTEs, joins, feature engineering, KPI design, demographic segmentation, geographic reporting.

- [GitHub Project](https://github.com/ziqixu22/flu-shot-analytics)
- [Interactive Tableau Dashboard](https://public.tableau.com/app/profile/ziqi.xu6990/viz/FluShotsDashboard_17311271789370/Dashboard1)

---

### 2. Customer Sales Analytics Dashboard

**Domain:** Business / Retail Analytics  
**Tools:** Tableau

**Problem:** Turn transaction-level sales data into an interpretable view of revenue concentration, customer segments, geography, time trends, and discount behavior.

Core revenue aggregation is

$$
R_g=\sum_{i\in g}T_i,
$$

with segment share

$$
\text{RevenueShare}_g=\frac{R_g}{\sum_h R_h}.
$$

The workbook includes total revenue, monthly revenue, age-wise sales, gender-wise sales, region-wise revenue share, state-level mapping, and a quantity-discount relationship view.

**What it demonstrates:** BI design, KPI aggregation, customer segmentation, geographic analysis, demographic analysis, business storytelling.

- [GitHub Project](https://github.com/ziqixu22/customer-sales-analytics)
- [Interactive Tableau Dashboard](https://public.tableau.com/views/Wisesalescustomeranalysis/CustomerAnalysis?:language=zh-CN&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

---

### 3. Netflix Global Content Analysis

**Domain:** Media Analytics  
**Tools:** Tableau

**Problem:** Make a large title-level catalog easier to understand by aggregating content geographically and by content type.

For country or segment $g$,

$$
N_g=\sum_i\mathbf{1}(i\in g),
$$

with relative catalog share

$$
\text{Share}_g=\frac{N_g}{\sum_h N_h}.
$$

**What it demonstrates:** geographic visualization, categorical aggregation, content analysis, interactive exploration, communication of large-scale distribution patterns.

- [GitHub Project](https://github.com/ziqixu22/netflix-global-content-analysis)
- [Interactive Tableau Dashboard](https://public.tableau.com/app/profile/ziqi.xu6990/viz/Netflixbycountry_17311283978990/Netflix)

---

### 4. Illinois Enrollment Demographics Dashboard

**Domain:** Education / Institutional Analytics  
**Tools:** Tableau

**Problem:** Distinguish absolute enrollment growth from changes in demographic composition over time.

For group $g$ in year $t$,

$$
T_t=\sum_g X_{t,g}
$$

and demographic share is

$$
S_{t,g}=\frac{X_{t,g}}{T_t}.
$$

The Tableau workbook implements the annual denominator with a FIXED LOD expression and uses counts, percentages, ranks, stacked areas, and gender trends to compare long-run changes.

**What it demonstrates:** longitudinal analysis, LOD expressions, percentage-of-total logic, rank-order analysis, demographic visualization.

- [GitHub Project](https://github.com/ziqixu22/illinois-enrollment-demographics)

---

## Why the Evaluation Approach Differs by Project

These are primarily analytics / BI projects rather than supervised machine-learning benchmarks. Therefore, evaluation emphasizes **data and KPI validity** rather than forcing inappropriate metrics such as RMSE or classification accuracy.

Typical checks include:

- denominators remain consistent across dashboard views
- totals reconcile across mutually exclusive groups
- shares remain in $[0,1]$
- cumulative metrics are monotonic where expected
- filters operate on the intended analytical population
- multi-membership dimensions are not incorrectly treated as mutually exclusive

This distinction matters in interviews: the evaluation method should match the analytical objective.

## Portfolio Comparison

| Project | Main Question | Data Grain | Core Method | Evaluation |
|---|---|---|---|---|
| Flu Shot Analytics | Who received a flu shot, and how does uptake vary? | Patient | SQL cohort + binary KPI | Cohort / denominator consistency |
| Customer Sales Analytics | Where does revenue come from? | Transaction | Aggregation + segmentation | KPI reconciliation |
| Netflix Global Content | How is catalog content distributed? | Title | Geographic / categorical aggregation | Count / share consistency |
| Illinois Enrollment | How has demographic composition changed? | Year × group | LOD + longitudinal analysis | Annual total / share reconciliation |

## Skills Demonstrated

Tableau · SQL · Business Intelligence · KPI Design · Cohort Analysis · Customer Segmentation · Geographic Visualization · Longitudinal Analysis · LOD Expressions · Data Storytelling

## Repository Philosophy

The individual repositories are intentionally kept separate so that each project can be reviewed as a self-contained case study. This landing page provides the high-level map; the linked repositories provide the technical depth.

Where exact numerical outcomes cannot be verified from plain-text repository artifacts, the project READMEs explicitly avoid inventing results and direct the reader to the Tableau workbook as the visual source of truth.