# -*- coding: utf-8 -*-

#import dash
#import dash_core_components as dcc
#import dash_html_components as html
#import plotly.express as px

import numpy as np
from flask import Flask, request,  render_template
import pickle



app = Flask(__name__)

cls_model = pickle.load(open('cmodel8.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

#import portpicker
#port = portpicker.pick_unused_port()
#from google.colab import output
#output.serve_kernel_port_as_window(port)

#from gevent.pywsgi import WSGIServer
#host='localhost'
#app_server = WSGIServer((host, port), app)
#app_server.serve_forever()

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #n12 = np.squeeze(np.asarray(n2))

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.asarray(int_features)]
    c = cls_model.predict(final_features)  

    output = c[0]
    
    # output = round(c[0], 2)
    # output = round(r[0], 2)
    
    
    if output == 1:
        return render_template('index.html', Disease_type='psoriasis  {}'.format(output))
    if output == 2:
        return render_template('index.html', Disease_type='seboreic dermatitis  {}'.format(output))
    if output == 3:
        return render_template('index.html', Disease_type='lichen planus  {}'.format(output))
    if output == 4:
        return render_template('index.html', Disease_type='pityriasis rosea  {}'.format(output))
    if output == 5:
        return render_template('index.html', Disease_type='cronic derma titis  {}'.format(output))
    else:
        return render_template('index.html', Disease_type='pityriasis rubra pilaris  {}'.format(output))

if __name__ == "__main__":
    #app.run_server(debug=True)
    
    app.run(debug=True, use_reloader=False)


    #app.run(debug= True)
    #app.debug = True
    #app.run()
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080)
    #