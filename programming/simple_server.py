from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/v1/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OK"})

@app.route('/v1/completions', methods=['POST'])
def completions():
    data = request.json
    # You can add your logic to handle the completion request here.
    # For example, return a mock response.
    response = {
        "choices": [
            {
                "text": "This is a mock completion response."
            }
        ]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
