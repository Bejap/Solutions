import json
import requests


class GetData:

    def __init__(self, base_url, endpoint, api_key):
        self.base_url = base_url

        self.endpoint = endpoint

        self.full_url = f"{self.base_url}{self.endpoint}"

        self.api_key = api_key

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self.response = requests.get(self.full_url, headers=self.headers)

        if self.response.status_code == 200:

            data = self.response.json()
            print("Data received:", json.dumps(data, indent=4))

        else:
            print(f'ERROR: {self.response.status_code} - {self.response.text}')


data = GetData("https://dmigw.govcloud.dk/v2/lightningdata", "/collections", "1eb30e5a-fd28-4df4-85ad-0d14b2ad81b9")
