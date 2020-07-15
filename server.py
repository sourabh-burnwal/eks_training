from flask import Flask, request, jsonify
import predict

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def run():
    data = request.get_json(force=True)
    input_dt = data['input']
    result =  predict.predict(input_dt)
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
