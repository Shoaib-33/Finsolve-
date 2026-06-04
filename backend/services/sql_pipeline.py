import re
import uuid
from dataclasses import dataclass
from typing import Any

from backend.services.cache import TTL_SQL_GEN, TTL_SQL_RESULT, get_json, set_json
from backend.services.sql import get_columns, run_sql


SQL_BLOCKLIST = [
    "alter", "attach", "create", "delete", "detach", "drop", "exec", "execute",
    "insert", "pragma", "replace", "truncate", "update", "vacuum",
]

MUTATING_INTENT_PATTERNS = [
    r"\bdelete\b",
    r"\bremove\b",
    r"\berase\b",
    r"\bdrop\b",
    r"\btruncate\b",
    r"\bupdate\b",
    r"\bmodify\b",
    r"\bchange\b",
    r"\binsert\b",
    r"\badd\b",
    r"\bcreate\b",
    r"\breplace\b",
]


@dataclass
class PendingSqlApproval:
    approval_id: str
    username: str
    role: str
    user_query: str
    sql_query: str


pending_sql_approvals: dict[str, PendingSqlApproval] = {}


def _strip_sql_markdown(text: str) -> str:
    sql = text.strip()
    sql = re.sub(r"^```(?:sql)?", "", sql, flags=re.IGNORECASE).strip()
    sql = re.sub(r"```$", "", sql).strip()
    return sql


def generate_sql(user_query: str, llm, sql_context: dict[str, Any] | None = None) -> str:
    validate_sql_intent(user_query)

    columns = get_columns()
    cached = get_json("sql_gen", user_query, columns, sql_context)
    if cached:
        return cached

    direct_sql = generate_contextual_sql(user_query, sql_context)
    if direct_sql:
        set_json("sql_gen", direct_sql, TTL_SQL_GEN, user_query, columns, sql_context)
        return direct_sql

    context_text = "None"
    if sql_context:
        context_text = (
            f"Previous user question: {sql_context.get('user_query')}\n"
            f"Previous SQL: {sql_context.get('sql_query')}\n"
            f"Previous result columns: {sql_context.get('columns')}\n"
            f"Previous result rows: {sql_context.get('rows')}"
        )

    prompt = f"""You are a schema-aware Text2SQL generator for SQLite.

Generate exactly one read-only SQL query for this schema:
Table: employees
Columns: {columns}

Previous SQL context:
{context_text}

Rules:
- Return only SQL, with no markdown or explanation.
- Use only the employees table.
- Generate a SELECT query only.
- Do not modify data.
- If the user asks to add, change, update, delete, remove, or mutate data, return UNSUPPORTED_REQUEST.
- Do not include comments.
- Prefer explicit column names instead of SELECT * unless the user asks for all fields.
- If the user asks who has the highest or lowest value for a field, include both full_name and that field in the SELECT list.
- For follow-up questions with pronouns like his, her, their, this, or that, resolve them using the previous SQL context.
- If the previous result identifies a person and the user asks for another field about that person, filter by that person's full_name or employee_id.

User question: {user_query}"""

    sql_query = validate_sql(_strip_sql_markdown(llm.invoke(prompt).content))
    set_json("sql_gen", sql_query, TTL_SQL_GEN, user_query, columns, sql_context)
    return sql_query


def generate_contextual_sql(user_query: str, sql_context: dict[str, Any] | None) -> str | None:
    if not sql_context:
        return None

    query_lower = user_query.lower()
    has_reference = re.search(r"\b(his|her|their|this|that|the same|previous)\b", query_lower)
    if not has_reference:
        return None

    target_column = None
    if re.search(r"\bsalary\b", query_lower):
        target_column = "salary"
    elif re.search(r"\bemail\b", query_lower):
        target_column = "email"
    elif re.search(r"\brole\b", query_lower):
        target_column = "role"
    elif re.search(r"\bdepartment\b", query_lower):
        target_column = "department"
    elif re.search(r"\blocation\b", query_lower):
        target_column = "location"

    if target_column is None:
        return None

    columns = sql_context.get("columns", [])
    rows = sql_context.get("rows", [])
    if not rows:
        return None

    first_row = rows[0]
    row_data = dict(zip(columns, first_row))
    employee_id = row_data.get("employee_id")
    full_name = row_data.get("full_name")

    if employee_id:
        return validate_sql(f"SELECT {target_column} FROM employees WHERE employee_id = '{_sql_literal(employee_id)}'")
    if full_name:
        return validate_sql(f"SELECT {target_column} FROM employees WHERE full_name = '{_sql_literal(full_name)}'")

    return None


def _sql_literal(value: Any) -> str:
    return str(value).replace("'", "''")


def validate_sql_intent(user_query: str) -> None:
    query_lower = user_query.lower()
    for pattern in MUTATING_INTENT_PATTERNS:
        if re.search(pattern, query_lower):
            raise PermissionError("This assistant only supports read-only SQL questions. Data changes are not allowed.")


def validate_sql(sql_query: str) -> str:
    sql = sql_query.strip()
    if sql.endswith(";"):
        sql = sql[:-1].strip()

    sql_lower = sql.lower()

    if not sql:
        raise PermissionError("Generated SQL is empty.")

    if ";" in sql:
        raise PermissionError("Only one SELECT statement is allowed.")

    if "--" in sql or "/*" in sql or "*/" in sql:
        raise PermissionError("SQL comments are not allowed.")

    if not re.match(r"^\s*select\b", sql_lower):
        raise PermissionError("Only SELECT queries are allowed.")

    blocked = [word for word in SQL_BLOCKLIST if re.search(rf"\b{re.escape(word)}\b", sql_lower)]
    if blocked:
        raise PermissionError(f"Unsafe SQL keyword blocked: {blocked[0]}.")

    if not re.search(r"\bfrom\s+employees\b", sql_lower):
        raise PermissionError("SQL must read from the employees table.")

    other_tables = re.findall(r"\b(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)\b", sql_lower)
    if any(table != "employees" for table in other_tables):
        raise PermissionError("SQL can only read from the employees table.")

    return sql


def request_sql_approval(user_query: str, sql_query: str, username: str, role: str) -> dict[str, Any]:
    validate_sql_intent(user_query)

    approval_id = str(uuid.uuid4())
    pending_sql_approvals[approval_id] = PendingSqlApproval(
        approval_id=approval_id,
        username=username,
        role=role,
        user_query=user_query,
        sql_query=sql_query,
    )
    return {
        "type": "sql_approval",
        "approval_id": approval_id,
        "query": sql_query,
        "answer": "SQL execution is pending approval.",
    }


def execute_approved_sql(approval_id: str, username: str, role: str) -> dict[str, Any]:
    pending = pending_sql_approvals.pop(approval_id, None)
    if pending is None:
        raise KeyError("Approval request not found or already handled.")

    if pending.username != username or pending.role != role:
        raise PermissionError("This approval request belongs to another session.")

    validate_sql_intent(pending.user_query)
    sql_query = validate_sql(pending.sql_query)
    result = get_json("sql_result", sql_query)
    if result is None:
        result = run_sql(sql_query)
        set_json("sql_result", result, TTL_SQL_RESULT, sql_query)

    response = format_sql_result(result, sql_query)
    response["user_query"] = pending.user_query
    return response


def reject_sql_approval(approval_id: str, username: str, role: str) -> dict[str, Any]:
    pending = pending_sql_approvals.pop(approval_id, None)
    if pending is None:
        raise KeyError("Approval request not found or already handled.")

    if pending.username != username or pending.role != role:
        raise PermissionError("This approval request belongs to another session.")

    return {"type": "text", "answer": "SQL execution cancelled.", "sources": [], "faithful": True}


def format_sql_result(result: dict[str, Any], sql_query: str) -> dict[str, Any]:
    return {
        "type": "table",
        "columns": result["columns"],
        "rows": result["rows"],
        "query": sql_query,
    }


def build_sql_context(user_query: str, response: dict[str, Any], max_rows: int = 5) -> dict[str, Any]:
    return {
        "user_query": user_query,
        "sql_query": response.get("query"),
        "columns": response.get("columns", []),
        "rows": response.get("rows", [])[:max_rows],
    }
