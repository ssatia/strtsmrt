from bs4 import BeautifulSoup
import requests
import datetime

FNAME = "snp500_formatted.txt"
stocks = []

def getNewsForDate(date):
    file = open('data/news/' + date.strftime('%Y-%m-%d') + '.csv', 'w')
    print('Getting news for ' + date.strftime('%Y-%m-%d'))
    for i in range(len(stocks)):
        query = 'http://www.reuters.com/finance/stocks/companyNews?symbol=' + stocks[i] + '&date=' + format(date.month, '02d') + format(date.day, '02d') + str(date.year)
        print('Getting news for ' + stocks[i])
        print(query)
        response = requests.get(query)
        soup = BeautifulSoup(response.text, "html.parser")
        divs = soup.findAll('div', {'class': 'feature'})
        print('Found ' + str(len(divs)) + ' articles.')

        if(len(divs) == 0):
            continue

        data = u''
        for div in divs:
            data = data.join(div.findAll(text=True))
        file.write(stocks[i] + ',' + data.encode('utf-8').replace('\n', ' '))
        file.write('\n')
    file.close()

def getNews():
    date = datetime.date(2013, 7, 27)
    endDate = datetime.date(2016, 10, 7)

    while(date <= endDate):
        getNewsForDate(date)
        date += datetime.timedelta(days=1)

def init():
    global stocks
    with open(FNAME) as f:
        stocks = f.readlines()
    for i in range(len(stocks)):
        stocks[i] = stocks[i].rstrip('\n')

    getNews()

init()
