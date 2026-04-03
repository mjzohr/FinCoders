import ssl
from urllib.request import urlopen, Request

import row
from bs4 import BeautifulSoup

# Disable SSL certificate verification (not recommended, but works for this example)
ssl._create_default_https_context = ssl._create_unverified_context

finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN', 'GOOGL', 'AAPL']

news_tables = {}

for ticker in tickers:
    url = finviz_url + ticker
    req = Request(url=url, headers={'user-agent': 'srikar'})
    response = urlopen(req)

    html = BeautifulSoup(response.read(),'lxml')
    news_table = html.find('table', class_='fullview-news-outer')
    news_tables[ticker] = news_table


"""
parsed_data=[]
for ticker, news_table in news_tables.items():
    for rows in news_table.find_all('tr'):
        title = row.a.get_text()
        date_data = row.td.text_split(' ')
"""



amzn_data=news_tables['AMZN']
goog_data=news_tables['GOOGL']
amzn_rows=amzn_data.find_all('tr')
goog_rows=goog_data.find_all('tr')

for index, row in enumerate(amzn_rows):
    a_tag = row.a
    if a_tag is not None:
        title = a_tag.text
        timestamp=row.td.text
    else:
        title = ""  # or some other default value
    print(timestamp + "" + title)