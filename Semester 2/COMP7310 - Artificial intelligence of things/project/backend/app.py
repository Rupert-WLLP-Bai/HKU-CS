from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO
import paho.mqtt.client as mqtt
import sqlite3
import json
import time

app = Flask(__name__)
socketio = SocketIO(app)

# MQTT 配置
MQTT_BROKER = "broker.emqx.io"  # Test MQTT broker
MQTT_TOPIC = "flask/esp32/bjh"
DB_FILE = "mqtt_messages.db"

# 初始化 SQLite 数据库
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS messages (
                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                          timestamp TEXT,
                          message TEXT)''')
        conn.commit()

# 保存消息到 SQLite
def save_to_db(message):
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO messages (timestamp, message) VALUES (?, ?)", (timestamp, message))
        conn.commit()

# 从 SQLite 读取消息
def get_messages():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT timestamp, message FROM messages ORDER BY id DESC LIMIT 20")
        return cursor.fetchall()

# MQTT 回调函数
def on_connect(client, userdata, flags, rc):
    print("Connected to MQTT Broker with code:", rc)
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    message = msg.payload.decode()
    print("Received:", message)
    save_to_db(message)  # 保存到 SQLite
    socketio.emit('mqtt_message', {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'data': message})

# 初始化 MQTT 客户端
mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect(MQTT_BROKER, 1883, 60)
mqtt_client.loop_start()  # 在后台运行 MQTT 循环

# Flask 路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/messages')
def messages():
    messages = get_messages()
    return jsonify([{'timestamp': msg[0], 'data': msg[1]} for msg in messages])

if __name__ == '__main__':
    init_db()  # 初始化数据库
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
