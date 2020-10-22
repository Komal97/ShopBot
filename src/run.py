from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse, Message, Body 
from ChatBotModel import ChatBotModel
from utils import fetch_reply

app = Flask(__name__)

@app.route("/sms", methods = ['POST'])
def sms_reply():

    # fetch the message
    message_received = request.values.get('Body', '').lower()
    phone_no = request.values.get('From', '').lower()

    # create replies
    model = ChatBotModel()
    if message_received.lower().__contains__('train model'):
        model.trainChatBotModel()
        reply = "Model training is done..."
    else:
        reply = model.predictChatBotModel(message_received)
        if reply.__contains__('rephrase'):
            new_reply = fetch_reply(message_received, phone_no)
            if not new_reply.__contains__('?'):
                reply = new_reply

    response = MessagingResponse()
    response.message(reply)

    return str(response)

if __name__ == "__main__":
    app.run(debug = True, port = 4000)