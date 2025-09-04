# app.py
"""
Project: HCP360 NL→SQL Runner (Streamlit)
Add-ons: Few-shot via FAISS + Dynamic Table Selection + CSV Table Descriptions (fixed path)
         Conversational follow-ups (question rewriter + context carryover)
         Chat-style UI (Streamlit chat messages)
         ➕ Visuals: interactive table, auto chart, CSV download
         ➕ Rich table_info context (columns/types/PK-FK/comments/sample values/row est.)
Version: 2.3.0 (FAISS + Rich Table Info + Token Safety)
Maintainer: Debankit
Python: 3.10+

Summary:
  Streamlit UI to:
    (1) auto-discover or accept a list of candidate tables,
    (2) load HUMAN table descriptions from a fixed CSV path and index them with schema in FAISS,
    (3) pick only the most relevant tables for the (possibly follow-up) question,
    (4) generate SQL via LangChain with few-shot examples,
    (5) execute via QuerySQLDataBaseTool,
    (6) produce a concise business-stakeholder NL answer,
    (7) maintain conversation so users can ask follow-ups "with respect to" prior answers,
    (8) present a chat-style interface (user and assistant bubbles),
    (9) show visuals (interactive table + best-guess chart) and allow CSV download,
   (10) include compact, rich table metadata to improve SQL quality without blowing context.

Requirements:
  pip install streamlit langchain langchain-community langchain-openai sqlalchemy psycopg2-binary python-dotenv faiss-cpu pandas altair
  Set OPENAI_API_KEY in env or via the sidebar.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Dict, List
import json
import hashlib

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from urllib.parse import quote_plus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain  # noqa: F401  (kept for reference)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

# NEW: for DataFrame fetch and charts
from sqlalchemy import create_engine, text, inspect
import altair as alt
from sqlalchemy.exc import OperationalError
from typing import Iterable, Tuple

# ---- Backwards-compatible examples import ----
try:
    # Preferred: your new structure
    from examples import SQL_EXAMPLES  # type: ignore
except Exception:  # pragma: no cover
    try:
        from examples import examples as SQL_EXAMPLES  # type: ignore
    except Exception:
        SQL_EXAMPLES = []  # last-resort fallback


# ---------------- Setup ----------------
load_dotenv()
st.set_page_config(page_title="HCP 360 Self Serve Agent", page_icon="🧠", layout="wide")

st.markdown("<h1 style='text-align: center;'>🧠 HCP 360 Self Serve Agent</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 16px;'>Ask in natural language. I’ll pick relevant tables, use your CSV summaries, generate SQL (with few-shot help), run it, and answer in plain English.</p>", unsafe_allow_html=True)


# ---------------- Constants ----------------
CSV_PATH = "hcp360_table_descriptions.csv"   # ← fixed path you provided
EXCEL_PATH = "HCP 360 data description.xlsx"  # optional Excel source
EX_PERSIST_DIR = "faiss_examples"            # (renamed for clarity)
EX_COLLECTION = "sql_fewshot_examples"       # unused with FAISS but kept for consistency in code comments
SCHEMA_PERSIST_DIR = "faiss_schema"          # (renamed for clarity)
SCHEMA_COLLECTION = "sql_schema_tables"      # unused with FAISS but kept for semantics

# ---- NEW: prompt safety budgets ----
MAX_RESULT_CHARS = 6000       # cap raw DB result string fed to NL rephraser
MAX_TABLE_NOTES_CHARS = 2000  # cap human CSV notes block
MAX_HISTORY_CHARS = 1200      # cap context sent to follow-up rewriter
MAX_EXAMPLE_CHARS = 1000      # cap each few-shot example (q+sql total)
MAX_TABLEINFO_CHARS = 6000    # cap the entire table_info block sent to the LLM
MAX_PER_TABLE_CHARS = 2000    # cap info per table

SAMPLE_ROWS_FOR_PROFILE = 5         # how many rows to peek for sample values in rich table info
SAMPLE_VALUES_PER_COLUMN = 3        # how many example values to show per column
INCLUDE_SAMPLE_VALUES = True        # flip False for leaner prompts


# ---------------- Helpers ----------------
def _cap(s: str, n: int) -> str:
    s = s or ""
    return (s[:n] + " …") if len(s) > n else s


def _sanitize_table_name(name: str) -> str:
    name = os.path.splitext(os.path.basename(name or "table"))[0]
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    if not name:
        name = "table"
    if name[0].isdigit():
        name = "t_" + name
    return name.lower()


def ingest_csvs_to_sqlite(files: Iterable, db_uri: str, if_exists: str = "replace") -> list[Tuple[str, int]]:
    """Load uploaded CSV files into SQLite tables. Returns list of (table_name, row_count)."""
    created: list[Tuple[str, int]] = []
    engine = create_engine(db_uri)
    for f in files:
        try:
            table = _sanitize_table_name(getattr(f, "name", "table"))
            df = pd.read_csv(f)
            df.to_sql(table, engine, if_exists=if_exists, index=False)
            created.append((table, len(df)))
        except Exception:
            continue
    return created


def create_demo_sqlite_schema(db_uri: str) -> list[str]:
    """Create small demo tables in SQLite for local testing. Returns list of table names created."""
    engine = create_engine(db_uri)
    tables: list[str] = []
    try:
        persona = pd.DataFrame([
            {"pres_eid": 101, "hcp_name": "Alice Martin", "specialty": "Cardiology", "state": "NY"},
            {"pres_eid": 102, "hcp_name": "Brian Chen", "specialty": "Oncology", "state": "CA"},
            {"pres_eid": 103, "hcp_name": "Carmen Diaz", "specialty": "Dermatology", "state": "TX"},
        ])
        persona.to_sql("hcp360_persona", engine, if_exists="replace", index=False)
        tables.append("hcp360_persona")

        eng = pd.DataFrame([
            {"pres_eid": 101, "channel": "email", "date": "2024-06-01", "impressions": 120, "clicks": 8},
            {"pres_eid": 101, "channel": "web",   "date": "2024-06-10", "impressions": 300, "clicks": 20},
            {"pres_eid": 102, "channel": "email", "date": "2024-06-03", "impressions": 200, "clicks": 14},
            {"pres_eid": 103, "channel": "web",   "date": "2024-06-02", "impressions": 150, "clicks": 5},
        ])
        eng.to_sql("hcp360_prsnl_engmnt", engine, if_exists="replace", index=False)
        tables.append("hcp360_prsnl_engmnt")
    except Exception:
        pass
    return tables

def clean_sql(s: str | None) -> str:
    """Extract a runnable single SQL statement from LLM output (strip code fences, comments, trailing semicolons)."""
    if not s:
        return ""
    m = re.search(r"```(?:sql)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    if m:
        s = m.group(1)
    else:
        lines = s.splitlines()
        for i, line in enumerate(lines):
            if line.strip().lower().startswith(("select", "with")):
                s = "\n".join(lines[i:])
                break
    s = re.sub(r"^```(?:sql)?\s*|\s*```$", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"--.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    return s.rstrip(" \t;")


def is_select_only(sql: str) -> bool:
    """Basic safety check: allow only SELECT/CTE."""
    return bool(sql) and re.match(r"^\s*(with|select)\b", sql.strip(), re.IGNORECASE) is not None


def load_table_descriptions_from_csv(path: str) -> Dict[str, str]:
    """
    Load CSV with columns: table, description
    Returns dict {table_name: description}. If not found or malformed, returns {}.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = cols.get("table") or cols.get("name") or cols.get("table_name")
    dcol = cols.get("description") or cols.get("desc")
    if not tcol or not dcol:
        return {}
    desc_map: Dict[str, str] = {}
    for _, row in df.iterrows():
        t = str(row[tcol]).strip()
        d = str(row[dcol]).strip()
        if t:
            desc_map[t] = d
    return desc_map


def get_desc_for(table: str, desc_map: Dict[str, str]) -> str:
    """Case-insensitive lookup of description for a table name."""
    if table in desc_map:
        return desc_map[table]
    lower_map = {k.lower(): v for k, v in desc_map.items()}
    return lower_map.get(table.lower(), "")


def load_table_descriptions_from_excel(path: str) -> Dict[str, str]:
    """
    Load Excel with columns: table, description (first sheet by default).
    Returns dict {table_name: description}. If not found or malformed, returns {}.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        df = pd.read_excel(path)
    except Exception:
        return {}
    cols = {c.lower().strip(): c for c in df.columns}
    tcol = cols.get("table") or cols.get("name") or cols.get("table_name")
    dcol = cols.get("description") or cols.get("desc")
    if not tcol or not dcol:
        return {}
    desc_map: Dict[str, str] = {}
    for _, row in df.iterrows():
        t = str(row[tcol]).strip()
        d = str(row[dcol]).strip()
        if t:
            desc_map[t] = d
    return desc_map


def load_table_descriptions(csv_path: str, excel_path: str) -> Dict[str, str]:
    """Load from both CSV and Excel, merge. Excel entries override CSV on conflicts."""
    csv_desc = load_table_descriptions_from_csv(csv_path) if csv_path else {}
    xls_desc = load_table_descriptions_from_excel(excel_path) if excel_path else {}
    if not csv_desc and not xls_desc:
        return {}
    # merge with Excel taking precedence
    merged = dict(csv_desc)
    merged.update(xls_desc)
    return merged


# ----------- NEW: DataFrame + Visual helpers -----------#
def try_fetch_dataframe(sql: str, db_uri: str, max_rows: int = 500) -> pd.DataFrame | None:
    """
    Return a pandas DataFrame for the SQL (best-effort).
    Caps rows for UI responsiveness. Returns None on failure.
    """
    try:
        engine = create_engine(db_uri)
        q = text(sql)
        with engine.connect() as conn:
            df = pd.read_sql_query(q, conn)
        if max_rows and len(df) > max_rows:
            df = df.head(max_rows)
        return df
    except Exception:
        return None


def _coerce_dates_inplace(df: pd.DataFrame) -> None:
    """Try to convert object columns that look like dates into datetimes (in-place, best-effort)."""
    for c in df.columns:
        if df[c].dtype == "object":
            try:
                converted = pd.to_datetime(df[c], errors="raise", utc=False, infer_datetime_format=True)
                if converted.notna().mean() > 0.7:
                    df[c] = converted
            except Exception:
                pass


def render_visuals(df: pd.DataFrame) -> None:
    """Render an interactive table and a best-guess chart."""
    st.caption("Visual preview")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Best-guess chart selection
    _coerce_dates_inplace(df)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    dt_cols  = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # Time series (datetime + numeric)
    if dt_cols and num_cols:
        x, y = dt_cols[0], num_cols[0]
        try:
            st.line_chart(df.set_index(x)[y])
            return
        except Exception:
            pass

    # Categorical + numeric: decide between pie and bar
    if num_cols and cat_cols:
        x, y = cat_cols[0], num_cols[0]
        top = df[[x, y]].dropna().head(50)

        try:
            if top[x].nunique() < 5:  # few categories → pie chart
                chart = (
                    alt.Chart(top)
                    .mark_arc()
                    .encode(theta=f"{y}:Q", color=f"{x}:N", tooltip=list(top.columns))
                    .properties(height=320)
                )
            else:  # many categories → bar chart
                chart = (
                    alt.Chart(top)
                    .mark_bar()
                    .encode(x=alt.X(f"{x}:N", sort="-y"), y=f"{y}:Q", tooltip=list(top.columns))
                    .properties(height=320)
                )
            st.altair_chart(chart, use_container_width=True)
            return
        except Exception:
            pass

    # Scatter (two numerics)
    if len(num_cols) >= 2:
        x, y = num_cols[0], num_cols[1]
        try:
            chart = (
                alt.Chart(df.dropna(subset=[x, y]).head(1000))
                .mark_circle(size=60, opacity=0.6)
                .encode(x=f"{x}:Q", y=f"{y}:Q", tooltip=list(df.columns))
                .properties(height=320)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
            return
        except Exception:
            pass

    # If no suitable chart, quietly skip


# ----------- Auto-summary (deterministic, DF-driven) -----------#
def _pick_columns_for_summary(df: pd.DataFrame) -> Tuple[str | None, str | None]:
    """Return (category_col, value_col) heuristically.
    Prefers common names; falls back to first object + first numeric.
    """
    if df is None or df.empty:
        return (None, None)
    # Identify numeric columns
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # Identify candidates for category
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    # Prefer well-known value columns
    value_prefs = [
        "call_count", "count", "hcp_count", "hcps", "engagements", "total", "cnt",
    ]
    value_col = None
    for v in value_prefs:
        for c in df.columns:
            if c.lower() == v:
                value_col = c
                break
        if value_col:
            break
    if not value_col and numeric_cols:
        value_col = numeric_cols[0]

    # Prefer well-known category columns
    cat_prefs = [
        "city", "month", "territory", "territory_name", "state", "category", "name",
    ]
    cat_col = None
    for v in cat_prefs:
        for c in df.columns:
            if c.lower() == v:
                cat_col = c
                break
        if cat_col:
            break
    if not cat_col and cat_cols:
        cat_col = cat_cols[0]

    return (cat_col, value_col)


def _auto_short_summary(df: pd.DataFrame, question: str | None = None) -> str | None:
    """Produce a concise, consistent summary directly from the DataFrame when suitable.
    Handles simple aggregations like counts by month/city/state. Returns None if not applicable.
    """
    if df is None or df.empty:
        return None
    cat_col, val_col = _pick_columns_for_summary(df)
    if not cat_col or not val_col:
        return None
    try:
        # Coerce value column to numeric for safe sorting
        vals = pd.to_numeric(df[val_col], errors="coerce")
        if vals.isna().all():
            return None
        # Work on a copy to avoid mutating session state
        tmp = df[[cat_col, val_col]].copy()
        tmp[val_col] = pd.to_numeric(tmp[val_col], errors="coerce")
        tmp = tmp.dropna(subset=[val_col])
        if tmp.empty:
            return None
        tmp = tmp.sort_values(val_col, ascending=False)
        top = tmp.head(3).values.tolist()
        # Build a compact line like: Top cities: X 402, Y 410, Z 429.
        # Pick a label based on cat_col
        label_map = {
            "city": "cities",
            "month": "months",
            "state": "states",
            "territory": "territories",
            "territory_name": "territories",
        }
        label = label_map.get(cat_col.lower(), cat_col)
        pieces = [f"{str(a)} {int(b)}" for a, b in top]
        if not pieces:
            return None
        if question and re.search(r"\bmonth\b", question.lower()) and cat_col.lower() == "month":
            prefix = "Monthly call counts"
        else:
            prefix = f"Top {label} by count"
        return f"{prefix}: " + ", ".join(pieces) + "."
    except Exception:
        return None


# ----------- Schema Introspection + Self-heal helpers -----------#
def build_columns_index(db_uri: str, schema: str | None, tables: List[str]) -> dict[str, set[str]]:
    """Return {table: set(column names in various cases)} for quick existence checks."""
    index: dict[str, set[str]] = {}
    try:
        engine = create_engine(db_uri)
        insp = inspect(engine)
        for t in tables:
            cols: list[dict] = []
            try:
                cols = insp.get_columns(t, schema=(schema or None))
            except Exception:
                # best-effort; skip on failure
                cols = []
            names = {c.get("name", "") for c in cols if c.get("name")}
            variants: set[str] = set()
            for n in names:
                variants.update({n, n.lower(), n.upper()})
            index[t] = variants
    except Exception:
        pass
    return index


def find_tables_with_column(index: dict[str, set[str]], col: str) -> List[str]:
    """Case-insensitive find of tables containing a column name."""
    if not col:
        return []
    needles = {col, col.lower(), col.upper()}
    out: List[str] = []
    for t, cols in index.items():
        if any(n in cols for n in needles):
            out.append(t)
    return out


def suggest_join_keys(index: dict[str, set[str]], tables: List[str]) -> List[str]:
    """Suggest plausible join keys by intersecting common columns and prioritizing typical ID fields."""
    if not tables:
        return []
    inter: set[str] | None = None
    for t in tables:
        cols = {c.lower() for c in index.get(t, set())}
        inter = cols if inter is None else inter & cols
    inter = inter or set()
    priority = [
        "pres_eid", "hcp_eid", "hcp_id", "prescriber_id", "npi", "eid", "id",
    ]
    ordered = [k for k in priority if k in inter]
    if not ordered:
        ordered = sorted(list(inter))[:3]
    return ordered


def find_columns_containing(index: dict[str, set[str]], words: List[str]) -> dict[str, List[str]]:
    """Return {table: [columns]} where any column name contains any of the words (case-insensitive)."""
    needles = {w.lower() for w in words if isinstance(w, str) and len(w) >= 3}
    out: dict[str, List[str]] = {}
    if not needles:
        return out
    for t, cols in index.items():
        hits: List[str] = []
        for c in cols:
            lc = c.lower()
            if any(n in lc for n in needles):
                hits.append(c)
        if hits:
            out[t] = sorted(list(set(hits)))
    return out


# ---------------- Few-shot Examples (FAISS) ----------------
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings()


def _extract_tables_from_sql(sql: str) -> set[str]:
    """Best-effort extraction of table names from FROM/JOIN clauses.
    Returns bare table names without schema/quotes.
    """
    tables: set[str] = set()
    if not sql:
        return tables
    try:
        pattern = re.compile(r"\b(from|join)\s+([a-zA-Z0-9_\.\"`]+)", re.IGNORECASE)
        for _, ident in pattern.findall(sql):
            # strip alias if present later
            ident = ident.strip()
            ident = ident.strip('`"')
            # skip subqueries
            if ident.startswith("("):
                continue
            # take last segment after schema dot if any
            name = ident.split(".")[-1].strip('`"')
            if name:
                tables.add(name)
    except Exception:
        pass
    return tables


def build_example_selector(top_k_limit: int, k: int = 3, allowed_tables: List[str] | None = None, dialect: str | None = None) -> SemanticSimilarityExampleSelector:
    allowed = {t.lower() for t in (allowed_tables or [])}
    formatted = []
    # Normalize dialect
    d = (dialect or "").lower()
    d = "postgresql" if "postgres" in d else ("sqlite" if "sqlite" in d else d)
    # Tokens that are Postgres-specific and often incompatible with SQLite
    pg_only_tokens = [
        r"NULLS\s+LAST",
        r"DATE_TRUNC\(",
        r"\bINTERVAL\b",
        r"STRING_AGG\(",
        r"::",  # cast
        r"DATE\s+'",  # DATE '2025-08-01'
        r"\bILIKE\b",
        r"GROUPING\s+SETS",
    ]
    for ex in SQL_EXAMPLES:
        q = ex.get("question") or ex.get("input", "")
        s = (ex.get("sql") or ex.get("query", "")).replace("{top_k}", str(top_k_limit))
        # Dialect gating via example metadata
        ex_dialects = ex.get("dialects")
        if ex_dialects and d:
            if d not in [str(x).lower() for x in ex_dialects]:
                continue
        # Heuristic: filter out PG-only syntax when running on SQLite
        if d == "sqlite":
            for tok in pg_only_tokens:
                import re as _re
                if _re.search(tok, s, flags=_re.IGNORECASE):
                    s = None
                    break
            if s is None:
                continue
        if not (q and s):
            continue
        # Filter out examples that reference tables not present in the DB
        if allowed:
            mentioned = {t.lower() for t in _extract_tables_from_sql(s)}
            # keep only if every mentioned table is in allowed set
            if mentioned and not mentioned.issubset(allowed):
                continue
        # trim example payload to keep token usage bounded
        q = _cap(q, MAX_EXAMPLE_CHARS // 2)
        s = _cap(s, MAX_EXAMPLE_CHARS // 2)
        formatted.append({"question": q, "sql": s})
    embeddings = get_embeddings()

    # Try to load existing FAISS index first
    vs = None
    if os.path.exists(EX_PERSIST_DIR):
        try:
            vs = FAISS.load_local(
                EX_PERSIST_DIR,
                embeddings,
                allow_dangerous_deserialization=True,
            )
        except Exception:
            shutil.rmtree(EX_PERSIST_DIR, ignore_errors=True)

    # Build fresh if needed
    if vs is None:
        texts = [f"{ex['question']}\n{ex['sql']}" for ex in formatted]
        vs = FAISS.from_texts(texts=texts, metadatas=formatted, embedding=embeddings)
        os.makedirs(EX_PERSIST_DIR, exist_ok=True)
        vs.save_local(EX_PERSIST_DIR)

    selector = SemanticSimilarityExampleSelector(
        vectorstore=vs,
        k=max(0, int(k)),
        input_keys=["question"],
    )
    return selector


example_prompt = PromptTemplate(
    input_variables=["question", "sql"],
    template=(
        "User question:\n{question}\n\n"
        "Target SQL:\n{sql}\n"
        "----\n"
    ),
)


def build_sql_prompt_with_examples(selector: SemanticSimilarityExampleSelector) -> FewShotPromptTemplate:
    prefix = (
        "You are a senior data engineer generating SQL for {dialect}.\n\n"
        "READ AND FOLLOW THESE RULES STRICTLY:\n"
        "1) Output ONLY one valid SQL statement. No backticks, no comments, no natural language, no explanations.\n"
        "2) Use ONLY tables and columns that appear in the provided table_info for the ALLOWED TABLES.\n"
        "3) Do NOT add a WHERE/HAVING/QUALIFY clause unless the user explicitly requested a filter in the question.\n"
        "3a) Never compare person names by string concatenation or text equality (e.g., CONCAT(first_name,' ',last_name)).\n"
        "3b) If a relationship exists via keys (e.g., pres_eid), rely on that key and omit any name-based filters.\n"
        "4) Prefer explicit JOINs with clear ON conditions; qualify column names when ambiguous.\n"
        "5) Do NOT invent columns, tables, or values. If something is not in table_info, do not reference it.\n"
        "6) Select only columns needed to answer the question (avoid SELECT * unless truly necessary).\n"
        "7) Respect the row cap: use LIMIT {top_k} when appropriate. Do not add ORDER BY unless the user asks for ordering.\n"
        "8) Use {dialect}-compatible syntax only (no vendor-specific functions not supported by {dialect}).\n"
        "9) For aggregations, include proper GROUP BY clauses for all non-aggregated selected columns.\n"
        "10) If a filter is requested with free text (e.g., name matching), use safe, schema-backed columns only; avoid brittle string concatenations unless present in table_info.\n"
        "11) When the question narrows a 'universe' or filter using one table (e.g., engagements), but requires attributes stored in another table (e.g., demographics), JOIN the relevant tables using keys in table_info (e.g., pres_eid) instead of assuming the attribute exists in the filtered table.\n"
        "12) Preserve exact table names from table_info (e.g., do not shorten 'hcp360_persona' to 'persona').\n"
        "13) If the question asks 'Which HCPs/physicians/doctors...', return one row per HCP (use the unique key such as pres_eid). Include the requested attributes via JOINs, but keep the aggregation grain at the HCP level.\n"
        "14) For month filters, prefer a standard inclusive-exclusive range using an actual date/datetime column from table_info (e.g., >= 2025-08-01 and < 2025-09-01). Do not invent vendor-specific date fields.\n"
        "15) If the concept 'calls' exists, filter using only clearly indicated columns in table_info (e.g., interaction_type/channel flags that denote calls). If no such column exists, count the relevant engagement rows without inventing columns.\n"
        "16) If the user requests only a time-bucketed count (e.g., 'month wise call count'), aggregate solely by that time bucket; DO NOT include extra dimensions (e.g., territory) unless explicitly requested. Order by the time bucket ascending.\n"
        "17) Do NOT use aggregate functions in the WHERE clause (e.g., MAX/SUM/COUNT); instead, use HAVING after GROUP BY or compute the aggregate in a subquery/CTE and join/filter on it.\n"
        "\n"
        "PROCESS TO FOLLOW WHEN CREATING THE QUERY:\n"
        "A) Identify the minimal set of ALLOWED TABLES needed to answer the question.\n"
        "B) From table_info, copy exact column names; do not guess or rename.\n"
        "C) Define JOINs using keys/relationships evident in table_info (e.g., p.pres_eid = s.pres_eid). Do not add text-based identity matches when a key exists.\n"
        "C1) Respect the requested grain (e.g., HCP-level when the question says 'Which physicians'); aggregate with GROUP BY at that grain only.\n"
        "D) If and only if the user explicitly asks for filters, add them under WHERE (or HAVING for post-aggregation).\n"
        "E) Add GROUP BY / aggregates only if required by the question.\n"
        "F) Add ORDER BY only if the user requests sorting (e.g., top N, highest/lowest, alphabetical).\n"
        "G) Add LIMIT {top_k} to cap result size when the question does not demand all rows.\n"
        "\n"
        "Table summaries (human-curated from CSV):\n"
        "{table_notes}\n\n"
        "Schema details and sample rows for the ALLOWED TABLES (use exact column names from here):\n"
        "{table_info}\n\n"
        "Here are relevant solved examples:\n\n"
    )
    suffix = (
        "\nNow write SQL for the new user question.\n"
        "New question:\n{question}\n"
    )
    return FewShotPromptTemplate(
        example_selector=selector,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["question", "table_info", "top_k", "dialect", "table_notes"],
    )


# ---------------- Schema Index (Dynamic Table Selection via FAISS) ----------------
def discover_tables(db: SQLDatabase, candidate_tables: List[str] | None) -> List[str]:
    """Return only tables that actually exist in the connected DB/schema.

    - If candidate_tables provided, intersect with live DB tables; if none remain, fall back to all tables.
    - If no candidates provided, return all usable tables.
    """
    try:
        actual = set(db.get_usable_table_names())
    except Exception:
        actual = set()
    if candidate_tables:
        filtered = [t for t in candidate_tables if t in actual]
        return sorted(filtered) if filtered else sorted(list(actual))
    return sorted(list(actual))


def _safe_conn_fingerprint(db_uri: str, schema: str | None, table_names: List[str]) -> str:
    """Create a reproducible fingerprint that ignores credentials but ties to DB host/db/schema and tables."""
    try:
        # crude parse to mask password while keeping host/db
        masked = re.sub(r"://([^:]+):([^@]+)@", r"://\\1:***@", db_uri)
    except Exception:
        masked = db_uri
    payload = json.dumps({
        "db": masked,
        "schema": schema or "",
        "tables": sorted(list(set(table_names))),
    }, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_or_load_schema_vs(db: SQLDatabase, table_names: List[str], desc_map: Dict[str, str], rebuild: bool = False, *, db_uri: str = "", schema: str | None = None):
    embeddings = get_embeddings()

    # Connection-aware cache fingerprint
    fp_new = _safe_conn_fingerprint(db_uri or "", schema, table_names)
    fp_file = os.path.join(SCHEMA_PERSIST_DIR, ".fingerprint")

    if rebuild and os.path.exists(SCHEMA_PERSIST_DIR):
        shutil.rmtree(SCHEMA_PERSIST_DIR, ignore_errors=True)

    # If a saved FAISS index exists, try to load it
    if os.path.exists(SCHEMA_PERSIST_DIR) and not rebuild:
        try:
            fp_old = ""
            if os.path.exists(fp_file):
                with open(fp_file, "r", encoding="utf-8") as f:
                    fp_old = (f.read() or "").strip()
            if fp_old == fp_new:
                return FAISS.load_local(
                    SCHEMA_PERSIST_DIR,
                    embeddings,
                    allow_dangerous_deserialization=True,
                )
            else:
                # fingerprint mismatch → rebuild
                shutil.rmtree(SCHEMA_PERSIST_DIR, ignore_errors=True)
        except Exception:
            shutil.rmtree(SCHEMA_PERSIST_DIR, ignore_errors=True)

    # Build fresh
    texts, metas = [], []
    for t in table_names:
        try:
            base_info = db.get_table_info(table_names=[t])
            # Prefer schema-qualified description if present, else bare table name
            key = f"{schema}.{t}" if schema else t
            human_desc = get_desc_for(key, desc_map) or get_desc_for(t, desc_map)
            doc_text = f"Table: {t}\nDescription: {human_desc}\n\n{base_info}"
            texts.append(doc_text)
            metas.append({"table": t})
        except Exception:
            continue

    vs = FAISS.from_texts(
        texts=texts or [""],
        metadatas=metas or None,
        embedding=embeddings,
    )
    os.makedirs(SCHEMA_PERSIST_DIR, exist_ok=True)
    vs.save_local(SCHEMA_PERSIST_DIR)
    try:
        with open(fp_file, "w", encoding="utf-8") as f:
            f.write(fp_new)
    except Exception:
        pass
    return vs


def check_connection(uri: str) -> None:
    try:
        engine = create_engine(uri)
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
    except OperationalError as e:
        msg = str(e.orig) if hasattr(e, "orig") else str(e)
        if "password authentication failed" in msg.lower():
            st.error("❌ Database login failed. Check user/password (and URL-encoding).")
        elif "does not exist" in msg.lower() and "database" in msg.lower():
            st.error("❌ Database name is wrong or missing in the URI.")
        else:
            st.error(f"❌ Could not connect to the database:\n{msg}")
        st.stop()


def select_relevant_tables(schema_vs, question: str, k_tables: int) -> List[str]:
    k_tables = max(1, int(k_tables))
    docs = schema_vs.similarity_search(question, k=k_tables)
    ordered: List[str] = []
    for d in docs:
        name = (d.metadata or {}).get("table")
        if name and name not in ordered:
            ordered.append(name)
    return ordered


def build_table_notes(selected_tables: List[str], desc_map: Dict[str, str]) -> str:
    lines = []
    for t in selected_tables:
        # Prefer schema-qualified description if present, else bare table name
        key = f"{schema}.{t}" if ("schema" in globals() and schema) else t
        desc = get_desc_for(key, desc_map) or get_desc_for(t, desc_map)
        if desc:
            lines.append(f"- {t}: {desc}")
        else:
            lines.append(f"- {t}: (no description provided)")
    return "\n".join(lines) if lines else "(no descriptions provided)"


# ---------------- Rich table info (compact Postgres metadata) ----------------
def _pg_estimated_rows(conn, schema: str, table: str) -> int | None:
    q = text("""
        SELECT CASE WHEN relpages = 0 THEN NULL
                    ELSE (reltuples/NULLIF(relpages,0))::float8 * relpages END AS est_rows
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema AND c.relname = :table
    """)
    try:
        r = conn.execute(q, {"schema": schema, "table": table}).fetchone()
        return int(r[0]) if r and r[0] is not None else None
    except Exception:
        return None


def _pg_table_comment(conn, schema: str, table: str) -> str:
    q = text("""
        SELECT obj_description(c.oid) AS comment
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = :schema AND c.relname = :table
    """)
    try:
        r = conn.execute(q, {"schema": schema, "table": table}).fetchone()
        return (r[0] or "").strip() if r and r[0] else ""
    except Exception:
        return ""


def _pg_column_comments(conn, schema: str, table: str) -> dict[str, str]:
    q = text("""
        SELECT a.attname AS col, pgd.description AS comment
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
        LEFT JOIN pg_description pgd ON pgd.objoid = c.oid AND pgd.objsubid = a.attnum
        WHERE n.nspname = :schema AND c.relname = :table
    """)
    out = {}
    try:
        for col, comment in conn.execute(q, {"schema": schema, "table": table}):
            if comment:
                out[col] = str(comment).strip()
    except Exception:
        pass
    return out


def _pg_primary_keys(conn, schema: str, table: str) -> list[str]:
    q = text("""
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON kcu.constraint_name = tc.constraint_name
         AND kcu.table_schema = tc.table_schema
        WHERE tc.table_schema = :schema AND tc.table_name = :table AND tc.constraint_type = 'PRIMARY KEY'
        ORDER BY kcu.ordinal_position
    """)
    try:
        return [r[0] for r in conn.execute(q, {"schema": schema, "table": table}).fetchall()]
    except Exception:
        return []


def _pg_foreign_keys(conn, schema: str, table: str) -> list[tuple[str, str, str]]:
    # returns [(local_col, ref_table, ref_col), ...]
    q = text("""
        SELECT
          kcu.column_name AS col,
          ccu.table_name  AS ref_table,
          ccu.column_name AS ref_col
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON kcu.constraint_name = tc.constraint_name
         AND kcu.table_schema = tc.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.table_schema = tc.table_schema
        WHERE tc.table_schema = :schema AND tc.table_name = :table AND tc.constraint_type = 'FOREIGN KEY'
        ORDER BY kcu.ordinal_position
    """)
    try:
        return [(r[0], r[1], r[2]) for r in conn.execute(q, {"schema": schema, "table": table}).fetchall()]
    except Exception:
        return []


def _format_table_summary(conn, engine, schema: str, table: str) -> str:
    insp = inspect(engine)
    cols = insp.get_columns(table, schema=schema)  # [{'name','type','nullable',...}]

    col_comments = _pg_column_comments(conn, schema, table)
    pks = _pg_primary_keys(conn, schema, table)
    fks = _pg_foreign_keys(conn, schema, table)
    tbl_comment = _pg_table_comment(conn, schema, table)
    est_rows = _pg_estimated_rows(conn, schema, table)

    # Sample a few rows (fast and capped)
    sample_df = None
    if INCLUDE_SAMPLE_VALUES and SAMPLE_ROWS_FOR_PROFILE > 0:
        try:
            sql = text(f'SELECT * FROM "{schema}"."{table}" LIMIT {SAMPLE_ROWS_FOR_PROFILE}')
            sample_df = pd.read_sql_query(sql, conn)
        except Exception:
            sample_df = None

    lines = []
    header = f"Table: {table}"
    if est_rows is not None:
        header += f"  (~{est_rows:,} rows est.)"
    lines.append(header)
    if tbl_comment:
        lines.append(f"Comment: {tbl_comment}")

    # Columns
    lines.append("Columns:")
    for c in cols:
        name = c.get("name")
        dtype = str(c.get("type"))
        nullable = "NULL" if c.get("nullable") else "NOT NULL"
        cmt = col_comments.get(name, "")
        col_line = f"  - {name}: {dtype}, {nullable}"
        if cmt:
            col_line += f" — {cmt}"
        # sample values (compact)
        if sample_df is not None and name in sample_df.columns:
            try:
                vals = (
                    sample_df[name]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )[:SAMPLE_VALUES_PER_COLUMN]
                if vals:
                    col_line += f" (eg: {', '.join(vals)})"
            except Exception:
                pass
        lines.append(col_line)

    # Keys/relationships
    if pks:
        lines.append("Primary key: " + ", ".join(pks))
    if fks:
        rels = [f"{lc} → {rt}.{rc}" for lc, rt, rc in fks]
        lines.append("Foreign keys: " + "; ".join(rels))

    summary = "\n".join(lines)
    return _cap(summary, MAX_PER_TABLE_CHARS)


def _format_table_summary_generic(conn, engine, table: str) -> str:
    insp = inspect(engine)
    cols = insp.get_columns(table)

    # Primary keys and foreign keys (best-effort for generic dialects)
    try:
        pk_info = insp.get_pk_constraint(table) or {}
        pks = pk_info.get("constrained_columns") or []
    except Exception:
        pks = []
    try:
        fk_info = insp.get_foreign_keys(table) or []
        fks = []
        for fk in fk_info:
            lc = ", ".join(fk.get("constrained_columns") or [])
            rt = fk.get("referred_table") or "?"
            rc = ", ".join(fk.get("referred_columns") or [])
            if lc and rt:
                fks.append((lc, rt, rc))
    except Exception:
        fks = []

    # Sample a few rows
    sample_df = None
    if INCLUDE_SAMPLE_VALUES and SAMPLE_ROWS_FOR_PROFILE > 0:
        try:
            sql = text(f'SELECT * FROM "{table}" LIMIT {SAMPLE_ROWS_FOR_PROFILE}')
            sample_df = pd.read_sql_query(sql, conn)
        except Exception:
            sample_df = None

    lines = [f"Table: {table}"]
    lines.append("Columns:")
    for c in cols:
        name = c.get("name")
        dtype = str(c.get("type"))
        nullable = "NULL" if c.get("nullable") else "NOT NULL"
        col_line = f"  - {name}: {dtype}, {nullable}"
        if sample_df is not None and name in sample_df.columns:
            try:
                vals = (
                    sample_df[name]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )[:SAMPLE_VALUES_PER_COLUMN]
                if vals:
                    col_line += f" (eg: {', '.join(vals)})"
            except Exception:
                pass
        lines.append(col_line)

    if pks:
        lines.append("Primary key: " + ", ".join(pks))
    if fks:
        rels = [f"{lc} → {rt}.{rc}" for lc, rt, rc in fks]
        lines.append("Foreign keys: " + "; ".join(rels))

    summary = "\n".join(lines)
    return _cap(summary, MAX_PER_TABLE_CHARS)


def build_rich_table_info(db_uri: str, schema: str | None, tables: list[str]) -> str:
    try:
        engine = create_engine(db_uri)
        dialect_name = engine.dialect.name.lower()
        with engine.connect() as conn:
            parts = []
            for t in tables:
                try:
                    if dialect_name == "postgresql":
                        use_schema = (schema or "public")
                        parts.append(_format_table_summary(conn, engine, use_schema, t))
                    else:
                        parts.append(_format_table_summary_generic(conn, engine, t))
                except Exception:
                    try:
                        db = SQLDatabase.from_uri(db_uri, schema=(schema or None), sample_rows_in_table_info=0)
                        parts.append(db.get_table_info(table_names=[t]))
                    except Exception:
                        parts.append(f"Table: {t}\n(no metadata available)")
            blob = "\n\n".join(parts)
            return _cap(blob, MAX_TABLEINFO_CHARS)
    except Exception:
        try:
            db = SQLDatabase.from_uri(db_uri, schema=(schema or None), sample_rows_in_table_info=0)
            return _cap(db.get_table_info(table_names=tables), MAX_TABLEINFO_CHARS)
        except Exception:
            return ""


# ---------------- NL Answer Prompt (business stakeholder) ----------------
answer_prompt = PromptTemplate.from_template(
    """You are given a user question, a SQL query (for context only), and the SQL result.
Write a concise, business-friendly insight that focuses only on what the result shows.

Strict constraints:
- Length: 2–3 short sentences, max ~280 characters total.
- No fluff, no restating the question, no headings or bullet lists.
- Do not mention SQL, queries, tables, or databases.
- If the result is numeric/aggregate, state the key number(s) and a plain-language interpretation.
- If the result is row-level data, give one short overall takeaway; optionally include a single concrete example — still within the length limit.
- If the SQL result is empty, do not infer or imagine any findings (the app will handle empty results elsewhere).

Question: {question}
(Context only — do not mention this): {query}
SQL Result: {result}

Answer:"""
)

def build_rephraser(llm: ChatOpenAI):
    return answer_prompt | llm | StrOutputParser()


# Enforce concise output in case the model gets verbose
def _enforce_concise(text: str, max_sentences: int = 3, max_chars: int = 280) -> str:
    if not text:
        return text
    s = str(text).strip()
    # Keep explicit empty-result message intact
    if s.lower().startswith("no relevant records were found"):
        return s
    # Normalize whitespace
    s = re.sub(r"\s+", " ", s)
    # Split into sentences
    parts = re.split(r"(?<=[.!?])\s+", s)
    s2 = " ".join(parts[:max_sentences]).strip()
    if len(s2) > max_chars:
        s2 = s2[: max(0, max_chars - 1)].rstrip() + "…"
    return s2


# ---------------- Follow-up Question Rewriter ----------------
followup_rewriter = PromptTemplate.from_template(
    """You are a helpful assistant for business stakeholders.
Your task: rewrite the user's new question into a clear, standalone business question, using the recent conversation for context.
Do not mention SQL, tables, or technical details. Resolve pronouns and references like "them", "that metric", or "the above".

Recent Q&A Context (most recent last):
{recent_qa}

New user question (may reference the above):
{new_question}

Rewrite the new question as a single, self-contained business question:
"""
)


# ---------------- Conversation state & helpers ----------------
if "history" not in st.session_state:
    # Each item: {"question","standalone_question","nl_answer","sql","result","selected_tables"}
    st.session_state["history"] = []

def last_turns(n: int = 3):
    h = st.session_state.get("history", [])
    return h[-n:] if h else []

def history_to_bullets(turns):
    if not turns:
        return "None."
    lines = []
    used = 0
    for i, t in enumerate(turns, 1):
        q = (t.get("question") or "").strip()
        a = (t.get("nl_answer") or "").strip()
        a = _cap(a, 300)  # tighter cap per answer
        chunk = f"{i}. Q: {q}\n   A: {a}\n"
        if used + len(chunk) > MAX_HISTORY_CHARS:
            break
        lines.append(chunk)
        used += len(chunk)
    return "".join(lines) if lines else "None."


# ---------------- Sidebar (CONFIG PANEL) ----------------
with st.sidebar:
    st.header("⚙️ Configuration")

    st.subheader("Database Source")
    use_server_db = st.toggle("Use server database (Postgres)", value=True)

    if use_server_db:
        st.caption("Connection (Neon)")
        user = st.text_input("User", value="neondb_owner")
        password='npg_QcHFVEf4o9uA'
        host = "ep-raspy-pond-a8p1p9je-pooler.eastus2.azure.neon.tech"
        database = st.text_input("Database", value="test")
        sslmode = "require"
        db_uri = f"postgresql+psycopg2://{user}:{quote_plus(password)}@{host}/{database}?sslmode=require"
        schema = st.text_input("Schema (optional)", value="dbo")
        dialect = "PostgreSQL"
        # st.code(db_uri.replace(quote_plus(password), "*****"), language="bash")
    else:
        st.caption("Local SQLite")
        sqlite_path = st.text_input("SQLite file path", value="local.db", help="Path to a .db or .sqlite file on this server.")
        db_uri = f"sqlite:///{sqlite_path}"
        schema = None
        dialect = "SQLite"

        with st.expander("Load data into SQLite"):
            st.caption("Upload CSV files and load them as tables into the local SQLite database.")
            uploaded = st.file_uploader("Upload one or more CSVs", type=["csv"], accept_multiple_files=True)
            mode = st.selectbox("Write mode", ["replace", "append"], index=0, help="Replace recreates tables. Append adds rows if table exists.")
            if st.button("Ingest uploaded CSVs", use_container_width=True, disabled=not uploaded):
                created = ingest_csvs_to_sqlite(uploaded or [], db_uri, if_exists=mode)
                if created:
                    for t, n in created:
                        st.success(f"Loaded {n} rows into table '{t}'.")
                else:
                    st.warning("No tables created. Ensure valid CSVs were uploaded.")

        if st.button("Create demo tables in SQLite", use_container_width=True):
            made = create_demo_sqlite_schema(db_uri)
            if made:
                st.success("Created demo tables: " + ", ".join(made))
            else:
                st.warning("Could not create demo tables (check write permissions).")

    st.subheader("Table Descriptions")
    desc_csv_path = st.text_input("CSV path", value=CSV_PATH)
    desc_xls_path = st.text_input("Excel path", value=EXCEL_PATH)
    exists_msg = []
    try:
        exists_msg.append(f"CSV: {'found' if os.path.exists(desc_csv_path) else 'missing'}")
    except Exception:
        exists_msg.append("CSV: unknown")
    try:
        exists_msg.append(f"Excel: {'found' if os.path.exists(desc_xls_path) else 'missing'}")
    except Exception:
        exists_msg.append("Excel: unknown")
    st.caption("; ".join(exists_msg))

    st.caption("Candidate tables (leave BLANK to auto-discover ALL tables from DB)")
    include_tables_raw = st.text_area(
        "One per line",
        value="",
        height=88,
        placeholder="hcp360_persona\nhcp360_prsnl_engmnt\n",
    )
    candidate_tables = [t.strip() for t in include_tables_raw.splitlines() if t.strip()]

    st.divider()
    # Default to larger-context models first
    model = st.selectbox("OpenAI Chat model", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    top_k = st.number_input("top_k (row cap per SELECT)", min_value=1, max_value=100000, value=1000, step=100)

    st.divider()
    st.subheader("🔎 Dynamic table selection")
    k_tables = st.slider("Max tables to include", 1, 2, 2, 1)
    rebuild_schema_index = st.button("Rebuild schema index")

    # NEW: toggle to include/exclude rich table info
    include_rich_table_info = st.checkbox("Include detailed table info in LLM context", value=True)

    st.divider()
    st.subheader("🧩 Few-shot examples")
    k_examples = st.slider("Examples to include (k)", min_value=0, max_value=8, value=5, step=1)
    rebuild_example_index = st.button("Rebuild example index")

    st.divider()
    if st.button("🗑️ Reset conversation"):
        st.session_state["history"] = []
        st.success("Conversation history cleared.")


# ---------------- Chat-style Main UI ----------------
# Lightweight run-time preferences (kept outside sidebar to avoid changing the config panel)
pref_cols = st.columns([1, 1, 1])
with pref_cols[0]:
    show_sql = st.toggle("Show generated SQL", value=True)
with pref_cols[1]:
    enforce_select_only = st.toggle("Enforce SELECT-only", value=True)
with pref_cols[2]:
    is_followup_default = st.toggle("Treat new questions as follow-ups", value=True)

# NEW: visuals toggle
show_visuals = st.toggle("Show visuals (table + chart)", value=True)

st.divider()

# Render past conversation as chat bubbles
for i, turn in enumerate(st.session_state["history"], 1):
    st.chat_message("user").write(turn.get("question", ""))

    with st.chat_message("assistant"):
        st.write(turn.get("nl_answer", ""))

        # Visuals for past turns (persist tables/charts)
        if show_visuals:
            hist_df = turn.get("df")
            if isinstance(hist_df, pd.DataFrame) and not hist_df.empty:
                with st.expander("Visuals", expanded=False):
                    render_visuals(hist_df)
                    csv_bytes = hist_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download these rows as CSV",
                        data=csv_bytes,
                        file_name=f"query_result_{i}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        key=f"dl_hist_{i}",
                    )

        # Nice collapsible details per assistant turn
        with st.expander("Details"):
            st.caption("Interpreted question")
            st.write(turn.get("standalone_question", ""))

            st.caption("Selected tables")
            sel = turn.get("selected_tables") or []
            st.write(", ".join(sel) if sel else "(none)")

            if show_sql:
                st.caption("Generated SQL")
                st.code(turn.get("sql", ""), language="sql")

            with st.expander("Raw DB result"):
                st.text(turn.get("result", ""))

st.divider()

# Single input for new user turns
prompt = st.chat_input("Ask a question about HCPs, publications, engagement, etc.")
if prompt:
    # Immediately render the user bubble
    st.chat_message("user").write(prompt)

    # ----- PROCESS THE QUESTION -----
    question = prompt
    if not os.environ.get("OPENAI_API_KEY"):
        with st.chat_message("assistant"):
            st.error("Missing OPENAI_API_KEY. Add it in the sidebar, environment, or Streamlit secrets.")
    elif not any([True for _ in [1]]):  # placeholder to mirror previous checks structure
        pass
    else:
        try:
            # 0) Load descriptions from CSV + Excel (Excel wins on conflicts)
            desc_map = load_table_descriptions(desc_csv_path, desc_xls_path)
            if not desc_map:
                st.warning(
                    "No table descriptions loaded from CSV/Excel. Proceeding without descriptions."
                )

            # 1) Init DB & LLM
            check_connection(db_uri)
            # IMPORTANT: disable sample rows to keep LangChain's schema lean
            db = SQLDatabase.from_uri(db_uri, schema=schema or None, sample_rows_in_table_info=0)
            llm = ChatOpenAI(model=model, temperature=temperature)

            # 2) Candidate tables
            all_candidates = discover_tables(db, [t.strip() for t in candidate_tables if t.strip()])
            if not all_candidates:
                with st.chat_message("assistant"):
                    st.error("No tables discovered. Provide candidate tables or verify schema/permissions.")
                st.stop()

            # 3) Build/load schema index (FAISS)
            if rebuild_schema_index and os.path.exists(SCHEMA_PERSIST_DIR):
                shutil.rmtree(SCHEMA_PERSIST_DIR, ignore_errors=True)
            schema_vs = build_or_load_schema_vs(db, all_candidates, desc_map, rebuild=False, db_uri=db_uri, schema=schema)

            # 4) Rewrite if follow-up
            standalone_question = question
            if is_followup_default and st.session_state["history"]:
                rewriter_llm = ChatOpenAI(model=model, temperature=0)
                standalone_question = (followup_rewriter | rewriter_llm | StrOutputParser()).invoke({
                    "recent_qa": history_to_bullets(last_turns(3)),
                    "new_question": question
                })

            # 5) Select tables (soft-bias prior)
            prior_tables: List[str] = []
            if is_followup_default and st.session_state["history"]:
                for t in last_turns(3):
                    prior_tables.extend(t.get("selected_tables", []))
            prior_tables = list(dict.fromkeys(prior_tables))

            if prior_tables:
                fresh_tables = select_relevant_tables(schema_vs, standalone_question, k_tables)
                merged: List[str] = []
                for t in prior_tables + fresh_tables:
                    if t not in merged:
                        merged.append(t)
                selected_tables = merged[:k_tables]
            else:
                selected_tables = select_relevant_tables(schema_vs, standalone_question, k_tables)

            if not selected_tables:
                with st.chat_message("assistant"):
                    st.error("Could not select relevant tables for the question.")
                st.stop()

            # Guard: if selection includes tables not present in the current DB, rebuild index and retry once
            try:
                actual_tables = set(db.get_usable_table_names())
            except Exception:
                actual_tables = set()
            if any(t not in actual_tables for t in selected_tables):
                schema_vs = build_or_load_schema_vs(db, list(actual_tables), desc_map, rebuild=True, db_uri=db_uri, schema=schema)
                selected_tables = select_relevant_tables(schema_vs, standalone_question, k_tables)
                if not selected_tables:
                    with st.chat_message("assistant"):
                        st.error("Table selection failed after index rebuild; verify DB connection and schema.")
                    st.stop()

            # 6) Few-shot + SQL generation
            if rebuild_example_index and os.path.exists(EX_PERSIST_DIR):
                shutil.rmtree(EX_PERSIST_DIR, ignore_errors=True)
            selector = build_example_selector(top_k_limit=int(top_k), k=k_examples, allowed_tables=all_candidates, dialect=dialect)
            custom_prompt = build_sql_prompt_with_examples(selector)

            # Dialect comes from the sidebar (Postgres vs SQLite)
            table_notes = build_table_notes(selected_tables, desc_map)
            table_notes = _cap(table_notes, MAX_TABLE_NOTES_CHARS)

            # Build detailed-but-compact table_info
            if include_rich_table_info:
                table_info = build_rich_table_info(db_uri, schema, selected_tables)
            else:
                # small fallback using LangChain (already no sample rows)
                table_info = _cap(db.get_table_info(table_names=selected_tables), MAX_TABLEINFO_CHARS)

            # Optional: add grain and metric hints for entity-focused questions to reduce misinterpretation
            grain_hint = ""
            try:
                qlow = (standalone_question or "").lower()
                if re.search(r"\bwhich\s+(physicians|doctors|hcps)\b", qlow):
                    grain_hint = (
                        "\nNote: Return one row per HCP (use the unique key such as pres_eid). "
                        "Include physician name if columns exist, plus specialty and territory. "
                        "Filter to the requested month using a valid date/datetime column from table_info. "
                        "Group at the HCP level and use HAVING COUNT(*) > N when counting interactions."
                    )
            except Exception:
                pass

            # Additional metric/date hints when question mentions "+N calls in <Month YYYY>"
            metric_hint = ""
            try:
                # Extract threshold N
                m_calls = re.search(r"\b(more than|greater than|>\s*)(\d+)\s+calls?\b", (standalone_question or ""), flags=re.IGNORECASE)
                n_calls = m_calls.group(2) if m_calls else None

                # Extract month-year like "August 2025"
                months = {
                    'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
                    'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
                }
                my = re.search(r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})\b", (standalone_question or ""), flags=re.IGNORECASE)
                from_to = None
                if my:
                    mon = months[my.group(1).lower()]
                    yr = int(my.group(2))
                    import datetime as _dt
                    start = _dt.date(yr, mon, 1)
                    if mon == 12:
                        end = _dt.date(yr + 1, 1, 1)
                    else:
                        end = _dt.date(yr, mon + 1, 1)
                    from_to = (start.isoformat(), end.isoformat())

                # Column hints from schema
                cols_index = build_columns_index(db_uri, schema or None, selected_tables)
                date_hits = find_columns_containing(cols_index, ["date", "dt", "time", "timestamp", "datetime"])  # table -> [cols]

                # Prefs for territory and names if present
                has_terr_name = any("territory_name" in {c.lower() for c in cols} for cols in cols_index.values())
                has_terr_id = any("territory_id" in {c.lower() for c in cols} for cols in cols_index.values())
                has_fname = any("first_name" in {c.lower() for c in cols} for cols in cols_index.values())
                has_lname = any("last_name" in {c.lower() for c in cols} for cols in cols_index.values())

                hint_parts = []
                if n_calls:
                    hint_parts.append(f"Include COUNT(*) AS call_count and HAVING call_count > {n_calls}.")
                if from_to:
                    hint_parts.append(f"For that month, filter with an inclusive-exclusive range: >= '{from_to[0]}' and < '{from_to[1]}'.")
                # List candidate date columns to avoid inventing vendor fields
                try:
                    if date_hits:
                        top = []
                        for t, cols in date_hits.items():
                            if t in selected_tables:
                                top.extend([f"{t}.{c}" for c in cols[:2]])
                        if top:
                            hint_parts.append("Use a real date/datetime column such as: " + ", ".join(top[:6]) + ".")
                except Exception:
                    pass
                if has_terr_name and has_terr_id:
                    hint_parts.append("Prefer territory name; if both name and id exist, use COALESCE(name, id) as territory.")
                if has_fname and has_lname:
                    hint_parts.append("If supported, include physician_name from first_name and last_name; otherwise keep separate.")

                if hint_parts:
                    metric_hint = "\n" + " ".join(hint_parts)
            except Exception:
                pass

            # Time-bucket hint for questions like "month wise call count" (no extra dimensions)
            time_hint = ""
            try:
                qlow2 = (standalone_question or "").lower()
                if re.search(r"\b(month[-\s]?wise|monthly|by month|per month)\b", qlow2):
                    parts = [
                        "Aggregate only by month; do not include other dimensions unless explicitly requested.",
                        "Order results by month ascending.",
                        "Use a valid date/datetime column from table_info for the month bucket.",
                        "For PostgreSQL use DATE_TRUNC('month', <date_col>), for SQLite use strftime('%Y-%m', <date_col>).",
                    ]
                    # Add concrete candidate date columns if we computed date_hits above
                    try:
                        if 'date_hits' in locals() and date_hits:
                            top = []
                            for t, cols in date_hits.items():
                                if t in selected_tables:
                                    top.extend([f"{t}.{c}" for c in cols[:2]])
                            if top:
                                parts.append("Candidate columns: " + ", ".join(top[:6]) + ".")
                    except Exception:
                        pass
                    time_hint = "\nNote: " + " ".join(parts)
            except Exception:
                pass

            # Weekly bucket hint for questions like "same week" / weekly comparisons
            weekly_hint = ""
            try:
                if re.search(r"\bsame\s+week\b|\bweekly\b", (standalone_question or ""), flags=re.IGNORECASE):
                    parts = [
                        "Derive a week bucket from a valid date/datetime column in table_info.",
                        "Group by pres_eid and the week bucket; use HAVING COUNT(DISTINCT <dimension>) to enforce 'both' conditions (e.g., two indications).",
                        "For PostgreSQL use DATE_TRUNC('week', <date_col>); for SQLite use strftime('%Y-%W', <date_col>).",
                        "Avoid aggregates in WHERE; use HAVING or subqueries for weekly filters.",
                    ]
                    # Add concrete candidate date columns if we computed date_hits above
                    try:
                        if 'date_hits' in locals() and date_hits:
                            top = []
                            for t, cols in date_hits.items():
                                if t in selected_tables:
                                    top.extend([f"{t}.{c}" for c in cols[:2]])
                            if top:
                                parts.append("Candidate date columns: " + ", ".join(top[:6]) + ".")
                    except Exception:
                        pass
                    weekly_hint = "\nNote: " + " ".join(parts)
            except Exception:
                pass

            # Context subset hint for phrases like "within those top 5 states"
            subset_hint = ""
            try:
                m_top = re.search(r"within\s+(?:those|the)\s+top\s*(\d+)\s*states", (standalone_question or ""), flags=re.IGNORECASE)
                if m_top:
                    n_states = m_top.group(1)
                    cols_index_all = build_columns_index(db_uri, schema or None, all_candidates)
                    state_hits = find_columns_containing(cols_index_all, ["state"])
                    hint = [
                        f"Compute the top {n_states} states by COUNT(DISTINCT pres_eid) and then restrict results to only those states.",
                        "Use an actual state column from table_info (e.g., from persona) and JOIN as needed.",
                        "Finally, aggregate by city and show cities with the highest counts across those states only.",
                    ]
                    try:
                        if state_hits:
                            top = []
                            for t, cols in state_hits.items():
                                if t in selected_tables:
                                    top.extend([f"{t}.{c}" for c in cols[:1]])
                            if top:
                                hint.append("Candidate state columns: " + ", ".join(top[:3]) + ".")
                    except Exception:
                        pass
                    subset_hint = "\nNote: " + " ".join(hint)
            except Exception:
                pass

            manual_inputs = {
                "question": f"{standalone_question}{grain_hint}{metric_hint}{time_hint}{weekly_hint}{subset_hint}",
                "table_info": table_info,
                "top_k": int(top_k),
                "dialect": dialect,
                "table_notes": table_notes,
            }
            sql_generator = custom_prompt | llm | StrOutputParser()
            sql_raw = sql_generator.invoke(manual_inputs)

            # 7) Safety & execution
            sql_to_run = clean_sql(sql_raw) or sql_raw
            if not sql_to_run:
                with st.chat_message("assistant"):
                    st.error("The LLM did not produce a valid SQL statement.")
                st.stop()
            if enforce_select_only and not is_select_only(sql_to_run):
                with st.chat_message("assistant"):
                    st.error("For safety, only SELECT/CTE queries are allowed.")
                    st.code(sql_to_run, language="sql")
                st.stop()

            execute_query = QuerySQLDataBaseTool(db=db)
            try:
                result_str = execute_query.invoke(sql_to_run)
                # NEW: try to fetch a tabular result for visuals (best-effort)
                df = try_fetch_dataframe(sql_to_run, db_uri)
            except Exception as exec_err:
                # Attempt a targeted self-heal for missing-column mistakes
                msg = str(exec_err)
                missing_col = None
                missing_table = None
                m = re.search(r"column\s+\"?([A-Za-z0-9_]+)\"?\s+does not exist", msg, flags=re.IGNORECASE)
                if not m:
                    m = re.search(r"column\s+([A-Za-z0-9_\.]+)\s+does not exist", msg, flags=re.IGNORECASE)
                if m:
                    missing_col = m.group(1).split(".")[-1]

                # Detect missing relation/table
                mt = re.search(r"relation\s+\"?([A-Za-z0-9_\.]+)\"?\s+does not exist", msg, flags=re.IGNORECASE)
                if not mt:
                    mt = re.search(r"table\s+\"?([A-Za-z0-9_\.]+)\"?\s+does not exist", msg, flags=re.IGNORECASE)
                if mt:
                    missing_table = mt.group(1)

                healed = False
                if missing_col:
                    cols_index = build_columns_index(db_uri, schema or None, all_candidates)
                    src_tables = find_tables_with_column(cols_index, missing_col)
                    # If we can locate the column in other tables, expand selection and retry with a hint
                    if src_tables:
                        join_hints = suggest_join_keys(cols_index, list(set(src_tables + selected_tables)))
                        selected_tables = list(dict.fromkeys(selected_tables + src_tables))

                        # Rebuild table_info with expanded tables
                        if include_rich_table_info:
                            table_info = build_rich_table_info(db_uri, schema, selected_tables)
                        else:
                            table_info = _cap(db.get_table_info(table_names=selected_tables), MAX_TABLEINFO_CHARS)

                        hint = (
                            f"Note: Column {missing_col} exists in: {', '.join(src_tables)}. "
                            f"JOIN tables using keys like: {', '.join(join_hints) if join_hints else 'refer table_info'}; "
                            f"do not reference {missing_col} from unrelated tables."
                        )
                        manual_inputs["question"] = f"{standalone_question}\n{hint}"
                        manual_inputs["table_info"] = table_info

                        sql_raw = sql_generator.invoke(manual_inputs)
                        sql_to_run = clean_sql(sql_raw) or sql_raw
                        if enforce_select_only and not is_select_only(sql_to_run):
                            raise exec_err

                        # Retry execution
                        result_str = execute_query.invoke(sql_to_run)
                        df = try_fetch_dataframe(sql_to_run, db_uri)
                        healed = True

                # If a table/relation was missing, force the model to use only allowed tables
                if (not healed) and missing_table:
                    allowed = ", ".join(selected_tables)
                    hint = (
                        f"Note: Table {missing_table} does not exist. Use only these ALLOWED TABLES: {allowed}. "
                        f"If needed, JOIN among them using keys from table_info."
                    )
                    manual_inputs["question"] = f"{standalone_question}\n{hint}"
                    sql_raw = sql_generator.invoke(manual_inputs)
                    sql_to_run = clean_sql(sql_raw) or sql_raw
                    if enforce_select_only and not is_select_only(sql_to_run):
                        raise exec_err
                    result_str = execute_query.invoke(sql_to_run)
                    df = try_fetch_dataframe(sql_to_run, db_uri)
                    healed = True

                if not healed:
                    # If we couldn't heal, re-raise original error to display
                    raise

            # 8) NL rephrase with empty-result guard (and optional zero-result retry)
            def _is_empty(res_str: str, frame: pd.DataFrame | None) -> bool:
                if isinstance(frame, pd.DataFrame) and frame.empty:
                    return True
                s = (res_str or "").strip().lower()
                return s in ("", "[]", "()", "none")

            if _is_empty(result_str, df):
                # Attempt one self-heal: identify potentially relevant columns based on question keywords
                # and ask the LLM to revise the SQL using those columns (if applicable).
                words = list({w.lower() for w in re.findall(r"[A-Za-z0-9]+", standalone_question) if len(w) >= 3})
                cols_index = build_columns_index(db_uri, schema or None, all_candidates)
                col_hits = find_columns_containing(cols_index, words)
                if col_hits:
                    hints = []
                    for t, cols in col_hits.items():
                        hints.append(f"{t}: {', '.join(cols[:6])}")
                    hint = (
                        "Note: The previous query returned zero rows. Based on the question keywords, "
                        "the following columns may be relevant for filtering: " + "; ".join(hints) + ". "
                        "Recheck filters using only ALLOWED TABLES and columns present in table_info."
                    )
                    manual_inputs["question"] = f"{standalone_question}\n{hint}"
                    sql_raw = sql_generator.invoke(manual_inputs)
                    sql_to_run = clean_sql(sql_raw) or sql_raw
                    if enforce_select_only and not is_select_only(sql_to_run):
                        # fall back to empty result handling
                        nl_answer = "No relevant records were found for this request."
                    else:
                        try:
                            result_str = execute_query.invoke(sql_to_run)
                            df = try_fetch_dataframe(sql_to_run, db_uri)
                        except Exception:
                            pass

                if _is_empty(result_str, df):
                    nl_answer = "No relevant records were found for this request."
                else:
                    safe_result_str = _cap(result_str, MAX_RESULT_CHARS)
                    rephrase_answer = build_rephraser(llm)
                    try:
                        nl_answer = rephrase_answer.invoke({
                            "question": standalone_question,
                            "query": sql_to_run,
                            "result": safe_result_str
                        })
                        nl_answer = _enforce_concise(nl_answer)
                    except Exception as e:
                        if "context length" in str(e).lower() or "context_length_exceeded" in str(e).lower():
                            nl_answer = (answer_prompt | ChatOpenAI(model=model, temperature=temperature) | StrOutputParser()).invoke({
                                "question": _cap(standalone_question, 400),
                                "query": _cap(sql_to_run, 1200),
                                "result": _cap(result_str, 3000),
                            })
                            nl_answer = _enforce_concise(nl_answer)
                        else:
                            raise
            else:
                safe_result_str = _cap(result_str, MAX_RESULT_CHARS)
                rephrase_answer = build_rephraser(llm)
                try:
                    nl_answer = rephrase_answer.invoke({
                        "question": standalone_question,
                        "query": sql_to_run,
                        "result": safe_result_str
                    })
                    nl_answer = _enforce_concise(nl_answer)
                except Exception as e:
                    if "context length" in str(e).lower() or "context_length_exceeded" in str(e).lower():
                        # Retry with minimal context
                        nl_answer = (answer_prompt | ChatOpenAI(model=model, temperature=temperature) | StrOutputParser()).invoke({
                            "question": _cap(standalone_question, 400),
                            "query": _cap(sql_to_run, 1200),
                            "result": _cap(result_str, 3000),
                        })
                        nl_answer = _enforce_concise(nl_answer)
                    else:
                        raise

            # 9) Render assistant bubble with answer + details
            with st.chat_message("assistant"):
                # Prefer deterministic short summary when we can infer it from the DataFrame
                auto_sum = _auto_short_summary(df, standalone_question)
                if auto_sum:
                    nl_answer = _enforce_concise(auto_sum)
                st.write(nl_answer)

                # NEW: Visuals section (optional)
                if show_visuals and df is not None and not df.empty:
                    with st.expander("Visuals", expanded=True):
                        render_visuals(df)
                        csv_bytes = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download these rows as CSV",
                            data=csv_bytes,
                            file_name="query_result.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                with st.expander("Details"):
                    st.caption("Interpreted question")
                    st.write(standalone_question)

                    st.caption("Selected tables")
                    st.write(", ".join(selected_tables))

                    if show_sql:
                        st.caption("Generated SQL")
                        st.code(sql_to_run, language="sql")

                    with st.expander("Raw DB result"):
                        st.text(result_str)

            # 10) Persist turn (store DataFrame for visuals to keep prior charts/tables visible)
            st.session_state["history"].append({
                "question": question,
                "standalone_question": standalone_question,
                "nl_answer": nl_answer,
                "sql": sql_to_run,
                "result": result_str,
                "selected_tables": selected_tables,
                "df": df,
            })

        except Exception as e:  # pragma: no cover
            with st.chat_message("assistant"):
                st.exception(e)
