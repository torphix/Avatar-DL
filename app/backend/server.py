from flask import Flask
from flask_cors import CORS
from models.avatar import Avatar
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['CORS_HEADERS'] = 'Content-Type'

socketio = SocketIO(app, cors_allowed_origins="http://localhost:8080",
                    logger=True, engineio_logger=True)

asr_kwargs = {}
tts_kwargs = {}
text_engine_kwargs = {}
video_engine_kwargs = {}

AVATAR = Avatar(
    asr_kwargs,
    tts_kwargs,
    text_engine_kwargs,
    video_engine_kwargs)

@socketio.on('connect')
def connect(auth):
    AVATAR.start()
    print('connected')

@socketio.on('audio')
def process_audio(audio_chunk):
    AVATAR.write_audio(audio_chunk)
    if AVATAR.output_queue.qsize() > 0:
        emit
if __name__ == '__main__':
    socketio.run(app)

