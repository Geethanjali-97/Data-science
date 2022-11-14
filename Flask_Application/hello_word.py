from flask import Flask ,render_template


app=Flask(__name__)

@app.route('/')
def home():
    return('Happy Sunday folks')
@app.route('/Login')
def Login():
    return('Please enter your login credentials')
@app.route('/Contact Details')
def Contact():
     return('Please enter your contact details')
app.run(debug=True)
