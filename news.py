import urllib2
import xmltodict

def getNewsForSymbol(symbol):
    file = urllib2.urlopen('http://feeds.finance.yahoo.com/rss/2.0/headline?s=' + symbol + '&region=US&lang=en-US')
    data = file.read()
    file.close()

    data = xmltodict.parse(data)
    data = data['rss']['channel']['item']
    headlines = []

    for i in range(len(data)):
        headlines.append(data[i]['title'])

    return headlines

def getNews():
    news = []
    filename = 'test_stocks.txt'
    with open(filename) as f:
        symbols = f.readlines()

    for i in range(len(symbols)):
        symbol = symbols[i].rstrip('\n')
        news.append((symbol, getNewsForSymbol(symbol)))

    return news

getNews()
