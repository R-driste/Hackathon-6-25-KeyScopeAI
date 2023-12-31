from flask import Flask, request, jsonify, redirect, url_for, render_template
from static.python.aiprocessing import aiprocess, clearlist

app = Flask(__name__)

@app.route("/info")
def info():
    return render_template("info.html")

@app.route("/")
def home():
    return render_template("index.html")

@app.errorhandler(404)
def page_not_found(error):
    return redirect("/")

@app.route('/useraudiodata', methods=['POST'])
def your_python_endpoint():
    data = request.get_json() 
    print(data)
    result = aiprocess(data)
    return jsonify(f"success, {result}")

if __name__ == "__main__":
    clearlist()
    app.run(debug=True)