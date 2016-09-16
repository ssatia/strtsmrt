import requests
from news import getNews

def analyzeText(text):
    result = requests.post('http://text-processing.com/api/sentiment/', data = {'text':text})
    return result.text

def analyzeSymbols():
    news = getNews()
    sentimentAnalysis = []

    for i in range(len(news)):
        sentimentAnalysis.append((news[i][0], analyzeText(news[i][1])))

    return sentimentAnalysis

# analyzeSymbols()
