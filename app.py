from flask import Flask, render_template, request, jsonify
import json
import requests  # Import the requests library

app = Flask(__name__)

def analyze_review_with_bert(review_text):
    """
    Sends the review_text to the external API for analysis and returns the result.
    """
    # Define the external API endpoint and required parameters
    api_url = "https://reviewsense.azurewebsites.net/api/analysereview"
    params = {
        'code': 'webapp',       # 'webapp' is API key or required code
        'review': review_text   # The review text to be analyzed
    }
    
    try:
        # Send a GET request to the external API with the parameters
        response = requests.get(api_url, params=params)
        response.raise_for_status()  # Raise an error for bad status codes
        
        # The API returns a JSON response
        analysis_result = response.json()
        return analysis_result
    except requests.exceptions.RequestException as e:
        # Handle any errors that occur during the request
        raise Exception(f"Error communicating with the analysis API: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    review_text = data.get('review', '').strip()
    
    if not review_text:
        return jsonify({'error': 'Review text is empty.'}), 400
    
    try:
        # Use the modified function to analyze the review via the external API
        analysis = analyze_review_with_bert(review_text)
        return jsonify({'analysis': analysis})
    except Exception as e:
        # Return a JSON response with the error message
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
