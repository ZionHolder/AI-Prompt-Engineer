from flask import Flask, request, jsonify, render_template
from model import prompt_eng_response
import time

app = Flask(__name__)

@app.route('/', methods=['GET'])

def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])

def generate():
    data = request.json
    question = data.get('question')
    prompt_style = data.get('prompt_style')

    if not question or not prompt_style:
        return jsonify({"error": "Missing question or prompt type selection"}), 400

    start_time = time.time()

    try:
        result = prompt_eng_response(question, prompt_style)
        result['duration'] = time.time() - start_time
        return jsonify(result)
    except Exception as e:
        return jsonify({"error" : str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
