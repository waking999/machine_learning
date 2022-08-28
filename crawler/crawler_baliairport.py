
import requests
from lxml import etree
import time

# https://www.airport-bali.com/bali-airlines
"""
home page
<div id="airlines-filter">
<div class="airline-item">
<a href="bali-airlines/aeromexico">
<img alt="AeroMexico">
"""

"""
each airline page
a href="/bali-departures-airline-aeromexico" style="text-decoration:underline;">Departures</a>
"""


"""
each airline departure page
<div class="flights-info">
<div class="flight-row">
<div class="flight-col flight-col__dest-term">
<b>Jakarta</b><span>(CGK)</span>

<div class="flight-col flight-col__flight">
<a href="/bali-flight-departure/AM7410">AM7410</a>
"""

destination = dict()
airline_dict = dict()

base_url = 'https://www.airport-bali.com'
url = 'https://www.airport-bali.com/bali-airlines'
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36'
    ,
    'cookies': 'IDE=AHWqTUlpS9IuYfeh8c7DFEoDRV5HYd7h3G1SIv6QRV2y2ckoFLQc6ZAUO-TexVFQlmM; DSID=ADtk6HOXkOVJODkIk9sb14s34OI-DL-nfDnw6OBeS3EsRQyxbTj_7SYORJDQ0fAsDrj0k6ZIJOM4kL3H7uuiKiHHXZjSlNGUznzPEzEGmiseKLLFw7Phl5WcFWY5vBVNm-WrhGrJo8wdcJykq9e7hlmDviLxtNp8UGVMSxG1npRAxQF2UpBZ9BNHMNhVcQmKk7JMWv9pFbmtIjuWa0vudGXnP81Pljg91u7n8XEK6QqF1mUb-AiLTGUimHnOBf9cuWvEPWySgjXyDh-sr1ynwZdJ016oxDjSe8NQa42D9Yw6Us1jWQPYmqc'
    , 'accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8'
}
response = requests.get(url=url, headers=header)

# airline home page
if response.status_code == 200:

    base_html = etree.HTML(response.text)
    airlines = base_html.xpath('//div[@class="airline-item__img"]/a')
    for airline in airlines:
        airline_href = airline.attrib['href']
        airline_name = airline.xpath('.//img')[0].attrib['alt']
        airline_dict[airline_name] = airline_href

        airline_url = base_url + airline_href
        time.sleep(2)
        airline_response = requests.get(url=airline_url, headers=header)

        # each airline page
        if airline_response.status_code == 200:

            airline_html = etree.HTML(airline_response.text)
            airline_departure_href = airline_html.xpath('//a[text()="Departures"]')[0].attrib['href']

            airline_departure_url = base_url + airline_departure_href
            time.sleep(2)
            airline_departure_response = requests.get(url=airline_departure_url, headers=header)

            # each airline departure page
            if airline_departure_response.status_code == 200:
                airline_departure_html = etree.HTML(airline_departure_response.text)

                airline_departure_rows = airline_departure_html.xpath('//div[@class="flight-row"]')

                for row in airline_departure_rows:
                    dest = row.xpath('.//div[@class="flight-col flight-col__dest-term"]')[0]
                    city = dest.xpath('.//b/text()')[0]
                    airport = dest.xpath('.//span/text()')[0]
                    dest_airport_name = city + '-' + airport

                    flight = row.xpath('.//div[@class="flight-col flight-col__flight"]/a/text()')[0]

                    if dest_airport_name in destination.keys():
                        dest_airlines = destination.get(dest_airport_name)
                        if airline_name in dest_airlines.keys():
                            flights = dest_airlines.get(airline_name)
                            flights.append(flight)

                        else:
                            flights = list()
                            flights.append(flight)
                            dest_airlines[airline_name] = flights
                    else:
                        dest_airlines = dict()
                        flights = list()
                        flights.append(flight)
                        dest_airlines[airline_name] = flights
                        destination[dest_airport_name] = dest_airlines

    print(destination)
