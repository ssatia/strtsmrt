import requests

FNAME = "snp500_formatted.txt"

def getHistData():
    DAY = '01'
    MONTH = '00' # month must be m - 1
    YEAR = '2013'

    with open(FNAME) as f:
        stocks = f.readlines()

    for i in range(len(stocks)):
        stock = stocks[i].rstrip('\n')
        query = 'http://ichart.finance.yahoo.com/table.csv?s=' + stock + '&a=' + MONTH + '&b=' + DAY + '&c=' + YEAR
        print 'Getting historical data for ' + stock
        print query
        response = requests.get(query)
        file = open('data/hsd/' + stock + '.csv', 'w')
        file.write(response.text)
        file.close()

getHistData()
