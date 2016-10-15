import requests
import json

def analyzeText(text):
    print "Performing sentiment analysis."
    jsonResponse = requests.post('http://text-processing.com/api/sentiment/', data = {'text': text})
    response = json.loads(jsonResponse.text)
    return response
