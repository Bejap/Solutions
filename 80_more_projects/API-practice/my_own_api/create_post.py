import requests

response = requests.post("http://127.0.0.1:5000/my_info",
                         json={'name': 'Allan'}
                         )

print(response.status_code)
print(response.json())
