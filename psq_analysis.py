import re
import io
import base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover - dependency may be missing/mis-matched at runtime
    sm = None


def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-")
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\,\.\:]+", "", s)
    return s


VALUE_MAP = {
    "strongly agree": 5,
    "agree": 4,
    "uncertain": 3,
    "neutral": 3,
    "disagree": 2,
    "strongly disagree": 1,
}


PSQ_ITEM_MAP = [
    ("doctors are good at explaining the medical test", "communication", False),
    ("my doctor’s office has everything needed", "technical_quality", False),
    ("my doctor's office has everything needed", "technical_quality", False),
    (
        "office has everything needed to provide complete medical care",
        "technical_quality",
        False,
    ),
    (
        "the medical care i have been receiving is just about perfect",
        "general_satisfaction",
        False,
    ),
    (
        "sometimes doctors make me wonder if their diagnosis is correct",
        "technical_quality",
        True,
    ),
    (
        "i feel confident that i can get the medical care i need without being set back financially",
        "financial_aspect",
        False,
    ),
    ("careful to check everything when treating and examining me", "technical_quality", False),
    (
        "i have to pay more for my medical care than that i can afford",
        "financial_aspect",
        True,
    ),
    ("i have easy access to medical specialists i need", "accessibility", False),
    ("people have to wait too long for emergency treatment", "accessibility", True),
    (
        "doctors act too business like and impersonal towards me",
        "interpersonal_manner",
        True,
    ),
    (
        "doctors act too busy and impersonal towards me",
        "interpersonal_manner",
        True,
    ),
    (
        "my doctor treats me in very friendly and courteous manner",
        "interpersonal_manner",
        False,
    ),
    (
        "those who provide my medical care sometimes hurry too much when they treat me",
        "time_with_doctor",
        True,
    ),
    ("doctors sometimes ignore what i tell them", "communication", True),
    (
        "i have some doubts about the ability of the doctors who treat me",
        "technical_quality",
        True,
    ),
    ("doctors usually spend plenty of time with me", "time_with_doctor", False),
    (
        "i find it hard to get an appointment for medical care right away",
        "accessibility",
        True,
    ),
    (
        "i am dissatisfied with something about the medical care i receive",
        "general_satisfaction",
        True,
    ),
    ("i am able to get medical care whenever i need it", "accessibility", False),
]


DOMAIN_LABELS = {
    "general_satisfaction": "General satisfaction",
    "technical_quality": "Technical quality",
    "interpersonal_manner": "Interpersonal manner",
    "communication": "Communication",
    "financial_aspect": "Financial aspect",
    "time_with_doctor": "Time spent with doctor",
    "accessibility": "Accessibility",
}


DEMO_FIELDS = [
    ("Age", "Age"),
    ("Sex", "Gender"),
    ("Marital Status", "Marital status"),
    ("Religion", "Religion"),
    ("Health Insurance", "Health insurance"),
    ("Educational Status", "Education level"),
    ("Distance from Hospital", "Distance from hospital"),
]


def _find_psq_columns(df: pd.DataFrame):
    norm_cols = {col: _norm(col) for col in df.columns}
    items = []
    taken = set()
    for pattern, domain, negative in PSQ_ITEM_MAP:
        pat = _norm(pattern)
        matched_col = None
        for col, ncol in norm_cols.items():
            if col in taken:
                continue
            if pat in ncol:
                matched_col = col
                break
        if matched_col:
            items.append(
                {
                    "pattern": pattern,
                    "col": matched_col,
                    "domain": domain,
                    "negative": negative,
                }
            )
            taken.add(matched_col)
    return items


def _to_score(val: str, negative: bool) -> float:
    if pd.isna(val):
        return np.nan
    s = _norm(str(val))
    score = VALUE_MAP.get(s, np.nan)
    if np.isnan(score):
        return np.nan
    if negative:
        score = 6 - score
    return score


def _table1_demographics(df: pd.DataFrame) -> str:
    total = len(df)
    lines = []
    lines.append("# Table 1. Sociodemographic Characteristics\n")
    lines.append("| Sociodemographic characteristics | Frequency (n) | Percentage (%) |")
    lines.append("|---------------------------------|---------------|----------------|")
    for field_key, field_title in DEMO_FIELDS:
        target = None
        for col in df.columns:
            if _norm(field_key) in _norm(col):
                target = col
                break
        if target is None:
            continue
        lines.append(f"| **{field_title}** | | |")
        vc = (
            df[target]
            .dropna()
            .astype(str)
            .apply(lambda x: re.sub(r"\s+", " ", x.strip()))
        )
        counts = vc.value_counts()
        for cat, n in counts.items():
            pct = (n / total) * 100 if total else 0.0
            lines.append(f"| {cat} | {n} | {pct:.1f}% |")
    lines.append(f"| **Total participants** | {total} | 100% |")
    return "\n".join(lines)


def _table2_item_distribution(df: pd.DataFrame, items_info: list) -> str:
    lines = []
    lines.append("# Table 2. Satisfaction of Patients (PSQ-II Items)\n")
    lines.append(
        "| Questions | No (Strongly disagree + Disagree) n (%) | Uncertain n (%) | Yes (Strongly agree + Agree) n (%) | Mean Score | Std. Dev. |"
    )
    lines.append(
        "|-----------|-----------------------------------------|-----------------|-----------------------------------|------------|-----------|"
    )
    for info in items_info:
        col = info["col"]
        neg = info["negative"]
        series = df[col].dropna().astype(str)
        series_n = len(series)
        if series_n == 0:
            no_n = unc_n = yes_n = 0
            mean_v = std_v = np.nan
        else:
            s_norm = series.apply(_norm)
            no_n = s_norm.isin(["strongly disagree", "disagree"]).sum()
            unc_n = s_norm.isin(["uncertain", "neutral"]).sum()
            yes_n = s_norm.isin(["agree", "strongly agree"]).sum()
            scores = s_norm.map(lambda v: VALUE_MAP.get(v, np.nan)).astype(float)
            if neg:
                scores = scores.map(lambda x: 6 - x if not np.isnan(x) else np.nan)
            mean_v = float(np.nanmean(scores)) if scores.notna().any() else np.nan
            std_v = (
                float(np.nanstd(scores, ddof=1)) if scores.notna().sum() > 1 else np.nan
            )

        def _pct(x):
            return f"{(x / series_n * 100):.1f}%" if series_n else "0.0%"

        question = info["pattern"].rstrip(".")
        lines.append(
            f"| {question} | {no_n} ({_pct(no_n)}) | {unc_n} ({_pct(unc_n)}) | {yes_n} ({_pct(yes_n)}) | {mean_v:.2f} | {std_v:.3f} |"
        )
    return "\n".join(lines)


def _compute_domain_scores_and_labels(df: pd.DataFrame, items_info: list):
    scored_cols = {}
    item_to_domain = {}
    for info in items_info:
        col = info["col"]
        neg = info["negative"]
        domain = info["domain"]
        item_to_domain[col] = domain
        scored_cols[col] = df[col].map(lambda v: _to_score(v, neg))
    scored_df = pd.DataFrame(scored_cols, index=df.index)

    domain_scores = {}
    domain_item_counts = {}
    for domain in DOMAIN_LABELS.keys():
        domain_items = [c for c, d in item_to_domain.items() if d == domain]
        if domain_items:
            domain_item_counts[domain] = len(domain_items)
            domain_scores[domain] = scored_df[domain_items].sum(axis=1, min_count=1)
    domain_scores_df = pd.DataFrame(domain_scores, index=df.index)

    labels = {}
    stats_rows = []
    for domain, scores in domain_scores_df.items():
        med = np.nanmedian(scores.values) if scores.notna().any() else np.nan
        lab = scores.apply(
            lambda x: 1 if (pd.notna(x) and x >= med) else (0 if pd.notna(x) else np.nan)
        )
        labels[domain] = lab
        mean_score = float(np.nanmean(scores)) if scores.notna().any() else np.nan
        std_score = (
            float(np.nanstd(scores, ddof=1)) if scores.notna().sum() > 1 else np.nan
        )
        n_valid = lab.notna().sum()
        pct_satisfied = (lab.sum() / n_valid * 100) if n_valid else np.nan
        avg_of_mean_score = mean_score / max(1, domain_item_counts.get(domain, 1))
        stats_rows.append(
            {
                "Satisfaction Domain": DOMAIN_LABELS[domain],
                "Mean of each domain": f"{mean_score:.2f}" if not np.isnan(mean_score) else "NA",
                "Std. Dev.": f"{std_score:.3f}" if not np.isnan(std_score) else "NA",
                "Average of mean score": f"{avg_of_mean_score:.2f}"
                if not np.isnan(avg_of_mean_score)
                else "NA",
                "% Satisfied": f"{pct_satisfied:.2f}%"
                if not np.isnan(pct_satisfied)
                else "NA",
            }
        )

    domain_stats_df = pd.DataFrame(stats_rows)
    return domain_scores_df, domain_stats_df, labels


def _table3_domains(domain_stats_df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Table 3. Satisfaction in Seven Dimensions of PSQ\n")
    lines.append(
        "| Satisfaction Domain  | Mean of each domain | Std. Dev. | Average of mean score | % Satisfied |"
    )
    lines.append(
        "|----------------------|---------------------|-----------|------------------------|-------------|"
    )
    for _, row in domain_stats_df.iterrows():
        lines.append(
            f"| {row['Satisfaction Domain']} | {row['Mean of each domain']} | {row['Std. Dev.']} | {row['Average of mean score']} | {row['% Satisfied']} |"
        )
    return "\n".join(lines)


def _pretty_var(v: str) -> str:
    mapping = {
        "Depression": "Experiencing depression (Yes)",
        "Female": "Female",
        "Edu_Primary": "Primary education",
        "Edu_Secondary": "Secondary education",
        "Edu_Religious": "Religious education",
        "Edu_BachelorsHigher": "Bachelor/Higher",
        "Insured": "With health insurance",
        "Age_31_40": "Age 31–40",
        "Age_41_50": "Age 41–50",
        "Age_51_60": "Age 51–60",
        "Age_60_plus": "Age 60+",
        "Dist_40min": "Travels 40 min",
        "Dist_50min": "Travels 50 min",
        "Dist_1hour": "Travels 1 hour",
        "Dist_>1hour": "More than 1 hour",
        "Occ_Retired": "Retired",
        "Occ_SelfEmployed": "Self-employed",
        "Occ_Student": "Student",
    }
    return mapping.get(v, v)


def _build_design_matrix(df: pd.DataFrame):
    def nval(colname, default=None):
        col = None
        for c in df.columns:
            if _norm(colname) in _norm(c):
                col = c
                break
        if col is None:
            return pd.Series([default] * len(df), index=df.index)
        return df[c]

    dep = (
        nval("You are experiencing depression in hospital", default=np.nan)
        .astype(str)
        .map(lambda x: 1 if _norm(x) == "yes" else (0 if _norm(x) == "no" else np.nan))
    )

    sex = nval("Sex", default=np.nan).astype(str)
    female = sex.map(
        lambda x: 1 if _norm(x) == "female" else (0 if _norm(x) == "male" else np.nan)
    )

    edu = nval("Educational Status", default=np.nan).astype(str).map(_norm)
    edu_sec = edu.map(lambda x: 1 if "secondary education" in x else 0 if not pd.isna(x) else np.nan)
    edu_pri = edu.map(lambda x: 1 if "primary education" in x else 0 if not pd.isna(x) else np.nan)
    edu_rel = edu.map(lambda x: 1 if "religious education" in x else 0 if not pd.isna(x) else np.nan)
    edu_bach = edu.map(
        lambda x: 1 if "bachelors" in x or "higher" in x else 0 if not pd.isna(x) else np.nan
    )

    ins = nval("Health Insurance", default=np.nan).astype(str)
    insured = ins.map(
        lambda x: 1 if _norm(x) == "yes" else (0 if _norm(x) == "no" else np.nan)
    )

    age = nval("Age", default=np.nan).astype(str).map(_norm)
    age_31_40 = age.map(lambda x: 1 if "31" in x and "40" in x else 0 if not pd.isna(x) else np.nan)
    age_41_50 = age.map(lambda x: 1 if "41" in x and "50" in x else 0 if not pd.isna(x) else np.nan)
    age_51_60 = age.map(lambda x: 1 if "51" in x and "60" in x else 0 if not pd.isna(x) else np.nan)
    age_60p = age.map(lambda x: 1 if "60 +" in x or "60+" in x else 0 if not pd.isna(x) else np.nan)

    dist = nval("Distance from Hospital", default=np.nan).astype(str).map(_norm)
    d_40 = dist.map(lambda x: 1 if x == "40 min" else 0 if not pd.isna(x) else np.nan)
    d_50 = dist.map(lambda x: 1 if x == "50 min" else 0 if not pd.isna(x) else np.nan)
    d_1h = dist.map(lambda x: 1 if x == "1 hour" else 0 if not pd.isna(x) else np.nan)
    d_1hp = dist.map(
        lambda x: 1 if "more then an hour" in x or "more than 1 hour" in x else 0 if not pd.isna(x) else np.nan
    )

    occ = nval("Occupation", default=np.nan).astype(str).map(_norm)
    occ_retired = occ.map(lambda x: 1 if "retired" in x else 0 if not pd.isna(x) else np.nan)
    occ_selfemp = occ.map(lambda x: 1 if "self employ" in x else 0 if not pd.isna(x) else np.nan)
    occ_student = occ.map(lambda x: 1 if "student" in x else 0 if not pd.isna(x) else np.nan)

    X = pd.DataFrame(
        {
            "Depression": dep,
            "Female": female,
            "Edu_Primary": edu_pri,
            "Edu_Secondary": edu_sec,
            "Edu_Religious": edu_rel,
            "Edu_BachelorsHigher": edu_bach,
            "Insured": insured,
            "Age_31_40": age_31_40,
            "Age_41_50": age_41_50,
            "Age_51_60": age_51_60,
            "Age_60_plus": age_60p,
            "Dist_40min": d_40,
            "Dist_50min": d_50,
            "Dist_1hour": d_1h,
            "Dist_>1hour": d_1hp,
            "Occ_Retired": occ_retired,
            "Occ_SelfEmployed": occ_selfemp,
            "Occ_Student": occ_student,
        },
        index=df.index,
    )
    return X


def _logistic_regression_tables(df: pd.DataFrame, labels: dict) -> pd.DataFrame:
    if sm is None:
        # statsmodels/scipy not available; skip modeling
        return pd.DataFrame(columns=["Satisfaction Domain","Variable","Adjusted Odds Ratio","P value"])
    X_all = _build_design_matrix(df)
    out_rows = []
    for domain, y in labels.items():
        y = y.copy()
        data = pd.concat([y.rename("y"), X_all], axis=1).dropna(subset=["y"])
        X = data.drop(columns=["y"]).copy()
        nunique = X.nunique(dropna=True)
        X = X.loc[:, nunique > 1]
        if X.empty or data["y"].nunique() < 2:
            continue
        X = sm.add_constant(X, has_constant="add")
        try:
            model = sm.Logit(data["y"], X).fit(disp=False)
            params = model.params
            pvals = model.pvalues
            for var in params.index:
                if var == "const":
                    continue
                OR = np.exp(params[var])
                p = pvals[var]
                out_rows.append(
                    {
                        "Satisfaction Domain": DOMAIN_LABELS[domain],
                        "Variable": _pretty_var(var),
                        "Adjusted Odds Ratio": f"{OR:.2f}",
                        "P value": f"{p:.3f}",
                    }
                )
        except Exception:
            continue
    if not out_rows:
        return pd.DataFrame(
            columns=[
                "Satisfaction Domain",
                "Variable",
                "Adjusted Odds Ratio",
                "P value",
            ]
        )
    return pd.DataFrame(out_rows)


def _table4_logistic(df: pd.DataFrame, labels: dict) -> str:
    res = _logistic_regression_tables(df, labels)
    sig = res[res["P value"].astype(float) < 0.05]
    view = sig if not sig.empty else res
    lines = []
    lines.append(
        "# Table 4. Association Between Independent Variables and 7 Domains of PSQ\n"
    )
    lines.append(
        "| Satisfaction Domain  | Variables         | Adjusted Odds Ratio | P value |"
    )
    lines.append("|----------------------|------------------|---------------------|---------|")
    for _, row in view.iterrows():
        lines.append(
            f"| {row['Satisfaction Domain']} | {row['Variable']} | {row['Adjusted Odds Ratio']} | {row['P value']} |"
        )
    lines.append("\nThreshold: 0.05")
    lines.append(
        "Reference groups: Age 20–30; Males; Unmarried; Muslims; Without health insurance; Illiterate; Travels 30 min"
    )
    return "\n".join(lines)


def _plot_domain_satisfaction(domain_stats_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    names = domain_stats_df["Satisfaction Domain"].tolist()
    pct = domain_stats_df["% Satisfied"].str.replace("%", "", regex=False).astype(float)
    ax.barh(names, pct, color="#4e79a7")
    ax.set_xlabel("% Satisfied")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def _methodology_text() -> str:
    return (
        "Methodology\n\n"
        "1) Categorized PSQ-II items into seven domains (General satisfaction, Technical quality, Interpersonal manner, "
        "Communication, Financial aspect, Time spent with doctor, Accessibility).\n"
        "2) Assigned Likert scores: Strongly agree=5, Agree=4, Neutral/Uncertain=3, Disagree=2, Strongly disagree=1; "
        "reverse-scored negatively worded items.\n"
        "3) Summed item scores within each domain per respondent.\n"
        "4) Applied a median cut-off to each domain score to create binary labels: 1=Satisfied (>= median), 0=Unsatisfied (< median).\n"
        "5) Fitted logistic regression models for each domain label against depression status and socio-economic characteristics "
        "(sex, education, health insurance, age group, distance to hospital, selected occupations), using 20–30 years, male, "
        "no insurance, illiterate, and 30 min travel as references.\n"
        "6) Reported Adjusted Odds Ratios (AOR) and p-values; produced summary tables and a satisfaction rate chart."
    )


def analyze_psq(df: pd.DataFrame):
    items_info = _find_psq_columns(df)
    table1 = _table1_demographics(df)
    table2 = _table2_item_distribution(df, items_info)
    _, domain_stats_df, labels = _compute_domain_scores_and_labels(df, items_info)
    table3 = _table3_domains(domain_stats_df)
    table4 = _table4_logistic(df, labels)
    chart_b64 = _plot_domain_satisfaction(domain_stats_df)

    segments = []
    segments.append({"type": "text", "content": table1})
    segments.append({"type": "text", "content": table2})
    segments.append({"type": "text", "content": table3})
    segments.append({"type": "image", "content": [chart_b64]})
    segments.append({"type": "text", "content": table4})
    segments.append({"type": "text", "content": _methodology_text()})

    meta = {"items_matched": len(items_info), "domains_covered": list(DOMAIN_LABELS.values())}
    return segments, meta
