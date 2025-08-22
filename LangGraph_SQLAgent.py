import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
from langchain_core.runnables.config import RunnableConfig
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START

load_dotenv()

DATABASE_URL= "postgresql+psycopg2://postgres:1234@localhost:5432/hcp360_poc"
engine = create_engine(
    DATABASE_URL,
    connect_args={"options": "-csearch_path=dbo"}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class AgentState(TypedDict):
    question: str
    sql_query: str
    query_result: str
    query_rows: list
    attempts: int
    relevance: str
    sql_error: bool


def get_database_schema(engine):
    inspector = inspect(engine)
    schema_str = []
    for table_name in inspector.get_table_names(schema="dbo"):
        schema_str.append(f"Table: {table_name}")
        cols = inspector.get_columns(table_name)
        pk = inspector.get_pk_constraint(table_name).get("constrained_columns", []) or []
        fk_info = inspector.get_foreign_keys(table_name)  # list of dicts

        # map col -> [fk targets]
        fk_map = {}
        for fk in fk_info:
            referred_table = fk.get("referred_table")
            referred_cols = fk.get("referred_columns", [])
            for c in fk.get("constrained_columns", []):
                targets = ", ".join(
                    f"{referred_table}.{rc}" for rc in referred_cols
                ) if referred_table and referred_cols else "unknown"
                fk_map.setdefault(c, []).append(targets)

        for column in cols:
            col_name = column["name"]
            col_type = str(column["type"])
            tags = []
            if col_name in pk:
                tags.append("Primary Key")
            if col_name in fk_map:
                targets = "; ".join(fk_map[col_name])
                tags.append(f"Foreign Key to {targets}")
            if tags:
                col_type = f"{col_type} ({', '.join(tags)})"
            schema_str.append(f"- {col_name}: {col_type}")
        schema_str.append("")  # blank line
    print("Retrieved database schema.")
    return "\n".join(schema_str)

class CheckRelevance(BaseModel):
    relevance: str = Field(
        description="Indicates whether the question is related to the database schema. 'relevant' or 'not_relevant'."
    )

def check_relevance(state: AgentState, config: RunnableConfig):
    question = state["question"]
    schema = get_database_schema(engine)
    print(f"Checking relevance of the question: {question}")
    system = """You are an assistant that determines whether a given question is related to the following database schema.

Schema:
{schema}

Respond with only "relevant" or "not_relevant".
""".format(schema=schema)
    human = f"Question: {question}"
    check_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", human),
        ]
    )
    llm = ChatOpenAI(temperature=0)
    structured_llm = llm.with_structured_output(CheckRelevance)
    relevance_checker = check_prompt | structured_llm
    relevance = relevance_checker.invoke({})
    state["relevance"] = relevance.relevance
    print(f"Relevance determined: {state['relevance']}")
    return state

class ConvertToSQL(BaseModel):
    sql_query: str = Field(
        description="The PostgreSql query corresponding to the user's natural language question."
    )

def convert_nl_to_sql(state: AgentState, config: RunnableConfig):
    question = state["question"]
    schema = get_database_schema(engine)
    print(f"Converting question to PostgreSql for user : {question}")
    system = """You are an assistant that converts natural language questions into SQL queries based on the following schema:

{schema}

⚠️ Important rules:
- Always wrap **all column names** in double quotes ("...") exactly as they appear in the schema.
- Do not lowercase or alter the identifiers. Preserve their exact spelling from the schema.
- Provide only the SQL query without any explanations.
- Alias columns appropriately to match the expected keys in the result (for example: "food"."name" AS food_name).
""".format(schema=schema)
    convert_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Question: {question}"),
        ]
    )
    llm = ChatOpenAI(temperature=0)
    structured_llm = llm.with_structured_output(ConvertToSQL)
    sql_generator = convert_prompt | structured_llm
    result = sql_generator.invoke({"question": question})
    state["sql_query"] = result.sql_query
    print(f"Generated SQL query: {state['sql_query']}")
    return state

def execute_sql(state: AgentState):
    sql_query = state["sql_query"].strip()
    session = SessionLocal()
    print(f"Executing SQL query: {sql_query}")
    try:
        result = session.execute(text(sql_query))
        if sql_query.lower().startswith("select"):
            rows = result.fetchall()
            columns = list(result.keys())
            if rows:
                # Use row._mapping for stable dict conversion in SQLAlchemy 2.x
                state["query_rows"] = [dict(r._mapping) for r in rows]
                # Pretty print
                header = ", ".join(columns)
                lines = []
                for r in state["query_rows"]:
                    parts = [f"{k}={r[k]}" for k in columns]
                    lines.append("; ".join(parts))
                formatted_result = f"{header}\n" + "\n".join(lines)
            else:
                state["query_rows"] = []
                formatted_result = "No results found."
            state["query_result"] = formatted_result
            state["sql_error"] = False

            print("SQL SELECT query executed successfully.")
        else:
            session.commit()
            state["query_result"] = "The action has been successfully completed."
            state["sql_error"] = False

            print("SQL command executed successfully.")
    except Exception as e:
        state["query_result"] = f"Error executing SQL query: {str(e)}"
        state["sql_error"] = True
        print(f"Error executing SQL query: {str(e)}")
    finally:
        session.close()
    return state

def generate_human_readable_answer(state: AgentState):
    sql = state["sql_query"]
    result = state["query_result"]
    query_rows = state.get("query_rows", [])
    sql_error = state.get("sql_error", False)
    print("Generating a human-readable answer.")
    system = """You are an assistant that converts SQL query results into clear, natural language responses without including any identifiers like order IDs. Start the response with a friendly greeting that includes the user's name."""
    if sql_error:
        generate_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", f"SQL Query:\n{sql}\n\nResult:\n{result}\n\nFormulate a clear and understandable error message in a single sentence,  informing them about the issue."),
            ]
        )
    elif sql.lower().startswith("select"):
        if not query_rows:
            generate_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    ("human", f"SQL Query:\n{sql}\n\nResult:\n{result}\n\nFormulate a clear and understandable answer to the original question in a single sentence."),
                ]
            )
        else:
            generate_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    ("human", f"SQL Query:\n{sql}\n\nResult:\n{result}\n\nSummarize the result in a single sentence suitable for a business user."),
                ]
            )
    else:
        generate_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", f"SQL Query:\n{sql}\n\nResult:\n{result}\n\nFormulate a clear and understandable confirmation message in a single sentence, confirming that the request has been successfully processed."),
            ]
        )
    llm = ChatOpenAI(temperature=0)
    human_response = generate_prompt | llm | StrOutputParser()
    answer = human_response.invoke({})
    state["query_result"] = answer
    print("Generated human-readable answer.")
    return state

class RewrittenQuestion(BaseModel):
    question: str = Field(description="The rewritten question.")

def regenerate_query(state: AgentState):
    question = state["question"]
    print("Regenerating the SQL query by rewriting the question.")
    system = """You are an assistant that reformulates an original question to enable more precise SQL queries. Ensure that all necessary details, such as table joins, are preserved to retrieve complete and accurate data.
    """
    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                f"Original Question: {question}\nReformulate the question to enable more precise SQL queries, ensuring all necessary details are preserved.",
            ),
        ]
    )
    llm = ChatOpenAI(temperature=0)
    structured_llm = llm.with_structured_output(RewrittenQuestion)
    rewriter = rewrite_prompt | structured_llm
    rewritten = rewriter.invoke({})
    state["question"] = rewritten.question
    state["attempts"] += 1
    print(f"Rewritten question: {state['question']}")
    return state

def generate_funny_response(state: AgentState):
    print("Generating a funny response for an unrelated question.")
    system = """You are a charming and funny assistant who responds in a playful manner."""
    human_message = "I can not help with that, but doesn't asking questions make you hungry? You can always order something delicious."
    funny_prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human_message)])
    llm = ChatOpenAI(temperature=0.7)
    funny_response = funny_prompt | llm | StrOutputParser()
    message = funny_response.invoke({})
    state["query_result"] = message
    print("Generated funny response.")
    return state

def end_max_iterations(state: AgentState):
    state["query_result"] = "Please try again."
    print("Maximum attempts reached. Ending the workflow.")
    return state

def relevance_router(state: AgentState):
    return "convert_to_sql" if state["relevance"].lower() == "relevant" else "generate_funny_response"

def check_attempts_router(state: AgentState):
    return "convert_to_sql" if state["attempts"] < 3 else "end_max_iterations"

def execute_sql_router(state: AgentState):
    return "generate_human_readable_answer" if not state.get("sql_error", False) else "regenerate_query"

workflow = StateGraph(AgentState)
workflow.add_node("check_relevance", check_relevance)
workflow.add_node("convert_to_sql", convert_nl_to_sql)
workflow.add_node("execute_sql", execute_sql)
workflow.add_node("generate_human_readable_answer", generate_human_readable_answer)
workflow.add_node("regenerate_query", regenerate_query)
workflow.add_node("generate_funny_response", generate_funny_response)
workflow.add_node("end_max_iterations", end_max_iterations)

workflow.add_edge(START, "check_relevance")
workflow.add_conditional_edges(
    "check_relevance",
    relevance_router,
    {
        "convert_to_sql": "convert_to_sql",
        "generate_funny_response": "generate_funny_response",
    },
)
workflow.add_edge("convert_to_sql", "execute_sql")
workflow.add_conditional_edges(
    "execute_sql",
    execute_sql_router,
    {
        "generate_human_readable_answer": "generate_human_readable_answer",
        "regenerate_query": "regenerate_query",
    },
)
workflow.add_conditional_edges(
    "regenerate_query",
    check_attempts_router,
    {
        "convert_to_sql": "convert_to_sql",
        "end_max_iterations": "end_max_iterations",  # FIXED key
    },
)
workflow.add_edge("generate_human_readable_answer", END)
workflow.add_edge("generate_funny_response", END)
workflow.add_edge("end_max_iterations", END)

app = workflow.compile()

# user_question_1 = "Show me how many HCPs are in each primary specialty."
# result_1 = app.invoke({"question": user_question_1, "attempts": 0})
# print("Result:", result_1["query_result"])
g = app.get_graph(xray=True)
import requests
mermaid_src = g.draw_mermaid()
resp = requests.post("https://kroki.io/mermaid/svg", data=mermaid_src.encode("utf-8"))
with open("workflow.svg", "wb") as f:
    f.write(resp.content)
user_question_2 = "Show the monthly count of email engagements by HCP for year 2024"
result_1 = app.invoke({"question": user_question_2, "attempts": 0})
print("Result:", result_1["query_result"])
