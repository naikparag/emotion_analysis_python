#!/usr/bin/env python3

from aiohttp import web
from multidict import MultiDict
import numpy as np
import pandas as pd

import librosa
import json

from keras.models import load_model

analysis_output_array = ['female_angry', 'female_calm', 'female_fearful', 'female_happy', 'female_sad', 'male_angry', 'male_calm', 'male_fearful', 'male_happy', 'male_sad']

async def handle(request):
    response_obj = { 'status' : 'success' }
    return web.Response(text=json.dumps(response_obj))

async def audioHandler(request):
    data = await request.post()
    audioData = data['audio']
    filename = audioData.filename
    print(filename)
    audioFile = data['audio'].file
    # await saveAudioFile(audioFile.read, filename)
    response_obj = { 'category' : processAudio(audioFile) }
    return web.Response(text=json.dumps(response_obj))

async def saveAudioFile(audioFile, name):
    savePath = 'data/'+ name
    savePathHandler = open(savePath,'w')
    savePathHandler.write(audioFile)
    savePathHandler.close()

def loadModel():
    # json_file = open('model.json', 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    # loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    # loaded_model.load_weights("saved_model/Emotion_Voice_Detection_Model.h5")
    # print("Loaded model from disk ---------------")
    
    loaded_model = load_model("saved_model/Emotion_Voice_Detection_Model.h5")
    print("Loaded model from disk ---------------")
    return loaded_model

def processAudio(audioFile):
    X, sample_rate = librosa.load('download.wav', res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5)
    print(X.shape)
    print(sample_rate)
    print(librosa.get_duration(y=X, sr=sample_rate))

    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20), axis=0)
    feature = mfccs

    livedf2= pd.DataFrame(data=feature)
    print(livedf2.shape)
    livedf2 = livedf2.stack().to_frame().T

    twodim= np.expand_dims(livedf2, axis=2)
    livepreds = loadModel().predict(twodim, batch_size=32, verbose=1)
    livepreds1 = livepreds.argmax(axis=1)
    pred_output_index = livepreds1.astype(int).flatten()[0]

    print(analysis_output_array[pred_output_index])
    return analysis_output_array[pred_output_index]

# WEB APPLICATION
# --------------------
app = web.Application()
app.router.add_get('/', handle)
app.router.add_post('/audio', audioHandler)

web.run_app(app, port=9999)
loadModel()