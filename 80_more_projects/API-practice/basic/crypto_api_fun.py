import requests

def get_crypto_price(crypto_name, against_currency):
    url = f"https://api.coingecko.com/api/v3/simple/price?ids={crypto_name}&vs_currencies={against_currency}"

    response = requests.get(url)
    data = response.json()

    print(data)

def get_crypto_price_against_diff_curr(crypto_name):
    list_against = ['dkk', 'eur', 'gbp', 'jpy', 'usd']
    for curr in list_against:
        get_crypto_price(crypto_name, curr)


get_crypto_price("bitcoin", "dkk")
get_crypto_price_against_diff_curr("doge")