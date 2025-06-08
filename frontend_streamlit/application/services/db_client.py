import requests


API_URL = "http://localhost:8000"


def register_user(name, email, password):
    try:
        response = requests.post(
            f"{API_URL}/register",
            params={"name": name, "email": email, "password": password}
        )
        if response.status_code == 200:
            return True, "✅ Registration successful!"
        else:
            return False, response.json().get("detail")
    except Exception as e:
        return False, str(e)


def login_user(email, password):
    try:
        response = requests.post(
            f"{API_URL}/login", json={"email": email, "password": password})
        if response.status_code == 200:
            return response.json().get("access_token"), None
        else:
            return None, response.json().get("detail")
    except Exception as e:
        return None, str(e)


def get_past_conversations(jwt_token, user_email):
    try:
        headers = {"Authorization": f"Bearer {jwt_token}"}
        params = {"email": user_email}
        response = requests.get(
            f"{API_URL}/messages", headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except:
        return []
