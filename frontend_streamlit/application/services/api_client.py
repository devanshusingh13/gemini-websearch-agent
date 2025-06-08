import requests
from typing import Dict, Any

BASE_URL = "http://localhost:8000"


def send_query(query: str, jwt_token: str) -> Dict[str, Any]:
    """Send query to backend with proper JWT authentication"""
    try:
        headers = {
            "Authorization": f"Bearer {jwt_token}",
            "Content-Type": "application/json"
        }
        # ✅ Log token before sending request
        print("Sending token:", jwt_token)

        response = requests.post(
            f"{BASE_URL}/generate",
            json={"query": query},
            headers=headers,
            timeout=30
        )

        # Debug authentication
        if response.status_code == 401:
            print(f"Auth Debug - Token: {jwt_token}")
            print(f"Auth Debug - Headers: {headers}")
            raise Exception("Authentication failed. Please login again.")

        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 401:
            raise Exception("Session expired. Please login again.")
        raise Exception(f"Failed to get response: {str(e)}")
