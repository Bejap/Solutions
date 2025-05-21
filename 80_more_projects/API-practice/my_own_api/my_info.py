from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/my_info', methods=['GET', 'POST'])
def hello():
    if request.method == 'POST':
        data = request.get_json()
        name = data.get('name', 'Anonymous')
        return jsonify(message=f"Hello {name}")
    return jsonify(message="Hello from the squat and deadlift guy")


if __name__ == '__main__':
    app.run(debug=True)