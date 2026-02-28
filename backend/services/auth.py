from retriever import users_db


def authenticate(username: str, password: str):
    user = users_db.get(username)
    if user and user["password"] == password:
        return {"username": username, "role": user["role"]}
    return None
