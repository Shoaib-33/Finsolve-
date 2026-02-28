import sqlite3
import pandas as pd

CSV_PATH = "resources/data/hr/hr_data.csv"
conn = sqlite3.connect("employees.db", check_same_thread=False)


def init_db():
    df = pd.read_csv(CSV_PATH)
    df.to_sql("employees", conn, if_exists="replace", index=False)
    print("✅ SQLite employees DB initialized.")


def get_columns():
    return pd.read_csv(CSV_PATH).columns.tolist()


def run_sql(sql_query: str):
    forbidden = ["drop", "delete", "insert", "update", "alter", "truncate", "exec", "--", ";--"]
    if any(word in sql_query.lower() for word in forbidden):
        raise PermissionError("Unsafe SQL query detected and blocked.")

    cursor = conn.cursor()
    cursor.execute(sql_query)
    rows = cursor.fetchall()
    col_names = [desc[0] for desc in cursor.description]
    return {"columns": col_names, "rows": [list(r) for r in rows]}
