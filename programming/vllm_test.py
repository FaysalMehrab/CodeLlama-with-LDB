import requests

try:
    response = requests.get("http://localhost:8000/v1/health")
    response.raise_for_status()
    print(response.status_code)
    print(response.json())
except requests.exceptions.RequestException as e:
    print(f"Connection error: {e}")
