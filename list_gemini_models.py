import requests



url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

response = requests.get(url)

print("Status Code:", response.status_code)
print(response.json())