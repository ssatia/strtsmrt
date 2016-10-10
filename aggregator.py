
FNAME = "snp500_formatted.txt"
stocks = []
data = []

def init():
    global stocks
    with open(FNAME) as f:
        stocks = f.readlines()
    for i in range(len(stocks)):
        stocks[i] = stocks[i].rstrip('\n')
