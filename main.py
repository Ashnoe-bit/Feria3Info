from flask import Flask, render_template, Response, jsonify, request
import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import platform
import socket

app = Flask(__name__)

# Inicializar insightface
face_analyzer = FaceAnalysis(providers=['CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Diccionario de recomendaciones
recomendaciones = {
    'enojado': ["Respira profundamente 3 veces", "Cuenta hasta 10", "Toma agua fria"],
    'enojada': ["Respira profundamente 3 veces", "Cuenta hasta 10", "Toma agua fria"],
    'miedoso': ["Respira lento: 4s inhala, 4s exhala", "Recuerda que estas a salvo", "Enfocate en el presente"],
    'miedosa': ["Respira lento: 4s inhala, 4s exhala", "Recuerda que estas a salvo", "Enfocate en el presente"],
    'feliz': ["¡Excelente! Disfruta el momento", "Comparte tu alegria", "Haz algo creativo"],
    'triste': ["Permitete sentir", "Habla con alguien", "Haz algo que te guste"],
    'sorprendido': ["Tómate un momento", "Respira y analiza", "Mantén la calma"],
    'sorprendida': ["Tómate un momento", "Respira y analiza", "Mantén la calma"],
    'neutral': ["Todo esta en calma", "Buen momento para concentrarte", "Organiza tus ideas"]
}

# Variables globales
resultado_actual = {
    'genero': 'N/A',
    'edad': 'N/A',
    'emocion': 'N/A',
    'recomendaciones': []
}

def obtener_ip_local():
    """Obtiene la IP local del servidor"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('192.168.1.1', 1))
        return s.getsockname()[0]
    except:
        return '127.0.0.1'
    finally:
        s.close()

def analizar_frame(frame):
    """Analiza un frame y devuelve los resultados"""
    global resultado_actual
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_analyzer.get(rgb)
    
    if len(faces) > 0:
        face = faces[0]
        
        # Género
        if hasattr(face, 'gender'):
            genero = 'Mujer' if face.gender == 0 else 'Hombre' if face.gender == 1 else 'N/A'
        else:
            genero = 'N/A'
        
        # Edad
        edad = int(face.age) if hasattr(face, 'age') and face.age else 'N/A'
        
        # Emoción
        if hasattr(face, 'emotion') and face.emotion:
            emocion_raw = max(face.emotion, key=face.emotion.get)
            emociones_map = {
                'angry': 'enojado' if genero == 'Hombre' else 'enojada',
                'disgust': 'disgustado' if genero == 'Hombre' else 'disgustada',
                'fear': 'miedoso' if genero == 'Hombre' else 'miedosa',
                'happy': 'feliz',
                'sad': 'triste',
                'surprise': 'sorprendido' if genero == 'Hombre' else 'sorprendida',
                'neutral': 'neutral'
            }
            emocion = emociones_map.get(emocion_raw, 'neutral')
        else:
            emocion = 'neutral'
        
        resultado_actual = {
            'genero': genero,
            'edad': edad,
            'emocion': emocion,
            'recomendaciones': recomendaciones.get(emocion, [])
        }
    
    return frame

def generar_frames():
    """Genera frames de video con análisis"""
    # Intentar abrir cámara (0 es la cámara por defecto)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        # Si no hay cámara, crear frame de error
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "Conecta una camara", (150, 240), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        while True:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analizar frame
        frame = analizar_frame(frame)
        
        # Agregar información en pantalla
        cv2.putText(frame, f"Genero: {resultado_actual['genero']}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Edad: {resultado_actual['edad']}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Emocion: {resultado_actual['emocion']}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Codificar frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/video')
def video():
    """Stream de video"""
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/resultado')
def resultado():
    """API para obtener resultados"""
    return jsonify(resultado_actual)

@app.route('/info')
def info():
    """Información del sistema"""
    return jsonify({
        'sistema': platform.system(),
        'python_version': platform.python_version(),
        'acceso': f"http://{obtener_ip_local()}:5000"
    })

if __name__ == '__main__':
    ip = obtener_ip_local()
    print("="*50)
    print("🎭 ANALIZADOR DE EMOCIONES")
    print("="*50)
    print(f"✅ Servidor iniciado")
    print(f"📱 Accede desde cualquier dispositivo:")
    print(f"   - Local: http://localhost:5000")
    print(f"   - Red:   http://{ip}:5000")
    print(f"💡 En tu celular/tablet, usa la IP de Red")
    print("="*50)
    
    app.run(host='0.0.0.0', port=5000, debug=False)