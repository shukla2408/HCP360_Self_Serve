# app.py
"""
Project: HCP360 NL→SQL Runner (Streamlit)
Add-ons: Few-shot via Chroma + Dynamic Table Selection + CSV Table Descriptions (fixed path)
         Conversational follow-ups (question rewriter + context carryover)
         Chat-style UI (Streamlit chat messages)
         ➕ Visuals: interactive table, auto chart, CSV download
Version: 2.2.0
Maintainer: Debankit
Python: 3.10+

Summary:
  Streamlit UI to:
    (1) auto-discover or accept a list of candidate tables,
    (2) load HUMAN table descriptions from a fixed CSV path and index them with schema in Chroma,
    (3) pick only the most relevant tables for the (possibly follow-up) question,
    (4) generate SQL via LangChain with few-shot examples,
    (5) execute via QuerySQLDataBaseTool,
    (6) produce a concise business-stakeholder NL answer,
    (7) maintain conversation so users can ask follow-ups "with respect to" prior answers,
    (8) present a chat-style interface (user and assistant bubbles),
    (9) show visuals (interactive table + best-guess chart) and allow CSV download.

Requirements:
  pip install streamlit langchain langchain-community langchain-openai sqlalchemy psycopg2-binary python-dotenv chromadb pandas altair
  Set OPENAI_API_KEY in env or via the sidebar.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain.chains import create_sql_query_chain  # noqa: F401  (kept for reference)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

# NEW: for DataFrame fetch and charts
from sqlalchemy import create_engine, text
import altair as alt

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
CSV_PATH = r"C:\HCP360\hcp360_table_descriptions.csv"   # ← fixed path you provided
EX_PERSIST_DIR = "chroma_examples"
EX_COLLECTION = "sql_fewshot_examples"
SCHEMA_PERSIST_DIR = "chroma_schema"
SCHEMA_COLLECTION = "sql_schema_tables"


# ---------------- Helpers ----------------
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


# ----------- NEW: DataFrame + Visual helpers -----------
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

# ---------------- Few-shot Examples (Chroma) ----------------
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings()


def build_example_selector(top_k_limit: int, k: int = 3) -> SemanticSimilarityExampleSelector:
    formatted = [
        {
            "question": ex.get("question") or ex.get("input", ""),
            "sql": (ex.get("sql") or ex.get("query", "")).replace("{top_k}", str(top_k_limit)),
        }
        for ex in SQL_EXAMPLES
        if (ex.get("question") or ex.get("input")) and (ex.get("sql") or ex.get("query"))
    ]
    embeddings = get_embeddings()
    vs = Chroma.from_texts(
        texts=[f"{ex['question']}\n{ex['sql']}" for ex in formatted],
        metadatas=formatted,
        embedding=embeddings,
        persist_directory=EX_PERSIST_DIR,
        collection_name=EX_COLLECTION,
    )
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
        "Target SQL (PostgreSQL):\n{sql}\n"
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
        "\n"
        "PROCESS TO FOLLOW WHEN CREATING THE QUERY:\n"
        "A) Identify the minimal set of ALLOWED TABLES needed to answer the question.\n"
        "B) From table_info, copy exact column names; do not guess or rename.\n"
        "C) Define JOINs using keys/relationships evident in table_info (e.g., p.pres_eid = s.pres_eid). Do not add text-based identity matches when a key exists.\n"
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


# ---------------- Schema Index (Dynamic Table Selection) ----------------
def discover_tables(db: SQLDatabase, candidate_tables: List[str] | None) -> List[str]:
    if candidate_tables:
        return candidate_tables
    try:
        return sorted(list(db.get_usable_table_names()))
    except Exception:
        return []


def build_or_load_schema_vs(db: SQLDatabase, table_names: List[str], desc_map: Dict[str, str], rebuild: bool = False) -> Chroma:
    embeddings = get_embeddings()
    if rebuild and os.path.exists(SCHEMA_PERSIST_DIR):
        shutil.rmtree(SCHEMA_PERSIST_DIR)

    if os.path.exists(SCHEMA_PERSIST_DIR) and not rebuild:
        try:
            return Chroma(
                collection_name=SCHEMA_COLLECTION,
                persist_directory=SCHEMA_PERSIST_DIR,
                embedding_function=embeddings,
            )
        except Exception:
            shutil.rmtree(SCHEMA_PERSIST_DIR)

    texts, metas = [], []
    for t in table_names:
        try:
            base_info = db.get_table_info(table_names=[t])
            human_desc = get_desc_for("dbo."+t, desc_map)
            doc_text = f"Table: {t}\nDescription: {human_desc}\n\n{base_info}"
            texts.append(doc_text)
            metas.append({"table": t})
        except Exception:
            continue

    vs = Chroma.from_texts(
        texts=texts or [""],
        embedding=embeddings,
        metadatas=metas or None,
        persist_directory=SCHEMA_PERSIST_DIR,
        collection_name=SCHEMA_COLLECTION,
    )
    return vs


def select_relevant_tables(schema_vs: Chroma, question: str, k_tables: int) -> List[str]:
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
        desc = get_desc_for("dbo."+t, desc_map)
        if desc:
            lines.append(f"- {t}: {desc}")
        else:
            lines.append(f"- {t}: (no description provided)")
    return "\n".join(lines) if lines else "(no descriptions provided)"


# ---------------- NL Answer Prompt (business stakeholder) ----------------
answer_prompt = PromptTemplate.from_template(
    """You are given a user question, a SQL query (for context only), and the SQL result. 
Write a clear, business-friendly answer that focuses only on the insights from the result.

Rules:
- Do not mention SQL, queries, tables, or databases. 
- Always frame the output as business insights for a stakeholder. 
- If the result is numeric/aggregate (counts, sums, averages, etc.), summarize the key finding in plain language. 
- If the result is row-level data (lists of people, products, transactions, etc.), summarize the overall insight first (e.g., number of rows), then list the details. 
- When listing rows:
    * Use the SELECT clause to infer column headers and their order. 
    * Present each row as a clean, human-readable record with labels. 
    * Skip null/None values — do not show them. 
- Keep formatting professional: start with a high-level summary, then show supporting details in bullet points or blocks. 
- Use business terminology from the question when possible. 
- 🚨 If the SQL Result is empty:
    * Do NOT invent, imagine, or infer any records. 
    * Simply respond with: "No relevant records were found for this request."

Question: {question}
(Context only — do not mention this): {query}
SQL Result: {result}

Answer:"""
)

def build_rephraser(llm: ChatOpenAI):
    return answer_prompt | llm | StrOutputParser()


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
    for i, t in enumerate(turns, 1):
        q = (t.get("question") or "").strip()
        a = (t.get("nl_answer") or "").strip()
        if len(a) > 500:
            a = a[:500] + " ..."
        lines.append(f"{i}. Q: {q}\n   A: {a}")
    return "\n".join(lines) if lines else "None."


# ---------------- Sidebar (UNCHANGED CONFIG PANEL) ----------------
with st.sidebar:
    st.header("⚙️ Configuration")

    default_uri = "postgresql+psycopg2://postgres:1234@localhost:5432/hcp360_poc"
    db_uri = st.text_input("SQLAlchemy DB URI", value=default_uri)
    schema = st.text_input("Schema (optional)", value="dbo")

    st.caption("Candidate tables (leave BLANK to auto-discover ALL tables from DB)")
    include_tables_raw = st.text_area(
        "One per line",
        value="",
        height=88,
        placeholder="hcp360_persona\nhcp360_persona_segment\nhcp360_persona_scientific_studies\nhcp360_prd_rtl_sls\nhcp360_prsnl_engmnt\n",
    )
    candidate_tables = [t.strip() for t in include_tables_raw.splitlines() if t.strip()]

    st.divider()
    model = st.selectbox("OpenAI Chat model", ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    top_k = st.number_input("top_k (row cap per SELECT)", min_value=1, max_value=100000, value=1000, step=100)

    st.divider()
    st.subheader("🔎 Dynamic table selection")
    k_tables = st.slider("Max tables to include", 1, 12, 4, 1)
    rebuild_schema_index = st.button("Rebuild schema index (re-embed tables + descriptions)")

    st.divider()
    st.subheader("🧩 Few-shot examples")
    k_examples = st.slider("Examples to include (k)", min_value=0, max_value=8, value=5, step=1)
    rebuild_example_index = st.button("Rebuild example index")

    st.divider()
    st.caption("Auth")
    api_key = st.text_input("OPENAI_API_KEY (optional if in env)", type="password")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    st.caption("Using table descriptions from CSV:")
    st.code(CSV_PATH, language="bash")

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
for turn in st.session_state["history"]:
    st.chat_message("user").write(turn.get("question", ""))

    with st.chat_message("assistant"):
        st.write(turn.get("nl_answer", ""))

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
            # 0) Load CSV descriptions
            desc_map = load_table_descriptions_from_csv(CSV_PATH)
            if not desc_map:
                st.warning(f"No table descriptions loaded from {CSV_PATH}. Proceeding without descriptions.")

            # 1) Init DB & LLM
            db = SQLDatabase.from_uri(db_uri, schema=schema or None)
            llm = ChatOpenAI(model=model, temperature=temperature)

            # 2) Candidate tables
            all_candidates = discover_tables(db, [t.strip() for t in candidate_tables if t.strip()])
            if not all_candidates:
                with st.chat_message("assistant"):
                    st.error("No tables discovered. Provide candidate tables or verify schema/permissions.")
                st.stop()

            # 3) Build/load schema index
            if rebuild_schema_index and os.path.exists(SCHEMA_PERSIST_DIR):
                shutil.rmtree(SCHEMA_PERSIST_DIR)
            schema_vs = build_or_load_schema_vs(db, all_candidates, desc_map, rebuild=False)

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

            # 6) Few-shot + SQL generation
            if rebuild_example_index and os.path.exists(EX_PERSIST_DIR):
                shutil.rmtree(EX_PERSIST_DIR)
            selector = build_example_selector(top_k_limit=int(top_k), k=k_examples)
            custom_prompt = build_sql_prompt_with_examples(selector)

            dialect = "PostgreSQL"
            table_notes = build_table_notes(selected_tables, desc_map)
            table_info = db.get_table_info(table_names=selected_tables)
            manual_inputs = {
                "question": standalone_question,
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
            result_str = execute_query.invoke(sql_to_run)

            # NEW: try to fetch a tabular result for visuals (best-effort)
            df = try_fetch_dataframe(sql_to_run, db_uri)

            # 8) NL rephrase
            rephrase_answer = build_rephraser(llm)
            nl_answer = rephrase_answer.invoke({
                "question": standalone_question,
                "query": sql_to_run,
                "result": result_str
            })

            # 9) Render assistant bubble with answer + details
            with st.chat_message("assistant"):
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

            # 10) Persist turn
            st.session_state["history"].append({
                "question": question,
                "standalone_question": standalone_question,
                "nl_answer": nl_answer,
                "sql": sql_to_run,
                "result": result_str,
                "selected_tables": selected_tables,
            })

        except Exception as e:  # pragma: no cover
            with st.chat_message("assistant"):
                st.exception(e)
