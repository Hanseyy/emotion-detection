from flask import Flask, render_template, Response, jsonify
import cv2
import sqlite3
import base64
import io
import time
from datetime import datetime
from collections import deque, defaultdict
from fer import FER
from matplotlib.figure import Figure

app = Flask(__name__)

# Initialize components
camera = cv2.VideoCapture(0)
detector = FER(mtcnn=True)
emotion_history = deque(maxlen=60)
emotion_counts = defaultdict(int)
total_detections = 0
current_emotion = "neutral"
current_conf = 0.0

EMOTION_COLORS = {
    'angry': '#FF4444', 'disgust': '#8B4513', 'fear': '#800080',
    'happy': '#FFD700', 'neutral': '#87CEEB', 'sad': '#4169E1', 'surprise': '#FF6347'
}

# SQLite DB setup
def init_db():
    conn = sqlite3.connect('emotions.db')
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS emotions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            emotion TEXT,
            confidence REAL
        )""")
    conn.commit()
    conn.close()

init_db()

def log_emotion(emotion, conf):
    conn = sqlite3.connect('emotions.db')
    c = conn.cursor()
    c.execute("INSERT INTO emotions (timestamp, emotion, confidence) VALUES (?, ?, ?)",
              (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), emotion, conf))
    conn.commit()
    conn.close()

def create_bar_chart(emotions):
    fig = Figure(facecolor='black')
    ax = fig.add_subplot(111, facecolor='black')
    bars = ax.bar(emotions.keys(), emotions.values(),
                  color=[EMOTION_COLORS.get(e, '#87CEEB') for e in emotions.keys()])
    ax.set_ylim([0, 1])
    ax.tick_params(colors='white')
    for bar, val in zip(bars, emotions.values()):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.01,
                f'{val:.2f}', ha='center', va='bottom', color='white')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='black')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_b64

def create_history_chart():
    if len(emotion_history) < 2:
        return None
    fig = Figure(facecolor='black')
    ax = fig.add_subplot(111, facecolor='black')
    counts = defaultdict(list)
    t_points = list(range(len(emotion_history)))
    for i, (emo, _) in enumerate(emotion_history):
        for e in EMOTION_COLORS.keys():
            counts[e].append(1 if emo == e else 0)
    for e, vals in counts.items():
        if any(vals):
            ax.plot(t_points, vals, label=e, color=EMOTION_COLORS[e])
    ax.legend(loc='upper right', fontsize=7)
    ax.tick_params(colors='white')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor='black')
    img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_b64

def gen_frames():
    global current_emotion, current_conf, total_detections
    frame_count = 0
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Resize frame to reduce processing load
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Detect only every 3rd frame for performance
        if frame_count % 3 == 0:
            try:
                results = detector.detect_emotions(small_frame)
            except Exception:
                results = []
        frame_count += 1

        if 'results' in locals() and results:
            emotions = results[0]["emotions"]
            dominant = max(emotions, key=emotions.get)
            conf = emotions[dominant]
            # Scale box coordinates to original frame size
            x, y, w, h = [int(coord * 2) for coord in results[0]["box"]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"{dominant.upper()} {int(conf * 100)}%", (x, y - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            emotion_history.append((dominant, conf))
            emotion_counts[dominant] += 1
            total_detections += 1
            current_emotion, current_conf = dominant, conf
            log_emotion(dominant, conf)
        else:
            cv2.putText(frame, "No Face Detected", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        ret, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/chart')
def chart():
    success, frame = camera.read()
    if success:
        res = detector.detect_emotions(frame)
        if res:
            emotions = res[0]["emotions"]
            chart_b64 = create_bar_chart(emotions)
            return jsonify({'chart': chart_b64, 'status':'ok'})
    return jsonify({'status':'no_data'})

@app.route('/history_chart')
def history_chart():
    data = create_history_chart()
    if data:
        return jsonify({'chart': data, 'status': 'ok'})
    return jsonify({'status': 'insufficient'})

@app.route('/stats')
def stats():
    if total_detections > 0:
        dom = max(emotion_counts, key=emotion_counts.get)
        percentages = {emo: (count/total_detections)*100 for emo, count in emotion_counts.items()}
        return jsonify({
            'current_emotion': current_emotion,
            'confidence': round(current_conf*100, 1),
            'dominant_emotion': dom,
            'percentages': percentages
        })
    return jsonify({'status': 'no_detections'})

@app.route('/detection')
def show_detections():
    conn = sqlite3.connect('emotions.db')
    c = conn.cursor()
    # Select the latest 100 detection records
    c.execute("SELECT * FROM emotions ORDER BY timestamp DESC LIMIT 100")
    rows = c.fetchall()
    conn.close()
    return render_template('detection.html', rows=rows)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)