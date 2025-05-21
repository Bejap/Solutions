import requests
import country_converter as coco

def guess_age(name):
    url = f"http://api.agify.io?name={name}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print("Error fetching data:", e)
        return

    print(f"Name: {data['name']}")
    print(f"Age: {data['age']}")
    print(f"Based on {data['count']} records")

def guess_gender(name):
    url = f"http://api.genderize.io?name={name}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print("Error fetching data:", e)
        return

    print(f"gender probability: {data['probability']}")
    print(f"gender: {data['gender']}")
    print(f"Based on {data['count']} records")

def guess_nationality(name):
    url = f"http://api.nationalize.io?name={name}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print("Error fetching data:", e)
        return
    sorted_list = sorted(data['country'], key=lambda x: x['probability'], reverse=True)
    if len(sorted_list) == 0:
        print("No data found")
    elif len(sorted_list) == 1:
        converted_country = coco.convert(names=sorted_list[0]['country_id'], to='name_long')
        print(f"this person is most likely to be from: {converted_country}, with a probability of {sorted_list[0]['probability']}")
    else:
        converted_country = coco.convert(names=sorted_list[0]['country_id'], to='name_long')
        print(f"this person is most likely to be from: {converted_country}, with a probability of {sorted_list[0]['probability']}")
        print(f"this person is secondly most likely to be from: {sorted_list[1]['country_id']}, with a probability of {sorted_list[1]['probability']}")


for i in range(4):
    print("-" * 30)
    name = input("Enter a name: ")
    guess_age(name)
    guess_gender(name)
    guess_nationality(name)
    print("-" * 30)

