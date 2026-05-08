import requests
import matplotlib.pyplot as plt
import time

topic = "Interfacial Engineering"
years = range(2020, 2027)
counts = []

with requests.session() as Sessions:
    for year in years:
        resp = Sessions.get(
            "https://api.openalex.org/works",
            params={
                "filter": f"publication_year:{year},title.search:{topic}",
                "per-page": 1  
            }
    ).json()

        if 'meta' in resp:
            counts.append(resp['meta']['count'])
        else:
            print(f"Unexpected response for {year}:", resp)
            counts.append(0)

        time.sleep(0.5)

plt.bar(years, counts)
plt.xlabel('Year')
plt.ylabel('Publications')
plt.title(f'Publications on "{topic}"')
plt.tight_layout()
plt.show()