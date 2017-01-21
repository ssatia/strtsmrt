from google.cloud import language

def analyzeText(text):
    print "Performing sentiment analysis."

    API_SIZE_LIMIT = 1000000
    text = text[:API_SIZE_LIMIT]
    language_client = language.Client()
    document = language_client.document_from_text(text)
    sentiment = document.analyze_sentiment()

    return sentiment
