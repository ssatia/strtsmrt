import requests
import json

def analyzeText(text):
    print "Performing sentiment analysis."
    POST_SIZE_LIMIT = 50000
    text = text[:POST_SIZE_LIMIT]
    jsonResponse = requests.post('http://text-processing.com/api/sentiment/', data = {'text': text})
    response = json.loads(jsonResponse.text)
    return response
