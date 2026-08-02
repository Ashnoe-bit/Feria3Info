from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from deepface import DeepFace
import platform
import socket
import threading

app = Flask(__name__)

# ── Recomendaciones por emoción ──
recomendaciones = {
    'feliz':        ["Excelente! Disfruta el momento", "Comparte tu alegria", "Haz algo creativo"],
    'triste':       ["Permitete sentir", "Habla con alguien de confianza", "Haz algo que te guste"],
    'enojado':      ["Respira profundamente 3 veces", "Cuenta hasta 10 lentamente", "Toma agua fria"],
    'enojada':      ["Respira profundamente 3 veces", "Cuenta hasta 10 lentamente", "Toma agua fria"],
    'estresado':    ["Para un momento y respira", "Haz una pausa de 5 minutos", "Escucha musica relajante"],
    'estresada':    ["Para un momento y respira", "Haz una pausa de 5 minutos", "Escucha musica relajante"],
    'miedoso':      ["Respira: 4s inhala, 4s exhala", "Recuerda que estas a salvo", "Enfocate en el presente"],
    'miedosa':      ["Respira: 4s inhala, 4s exhala", "Recuerda que estas a salvo", "Enfocate en el presente"],
    'sorprendido':  ["Tomaté un momento", "Respira y analiza la situacion", "Manten la calma"],
    'sorprendida':  ["Tomaté un momento", "Respira y analiza la situacion", "Manten la calma"],
    'disgustado':   ["Alejate de lo que te molesta", "Respira hondo", "Busca algo positivo"],
    'disgustada':   ["Alejate de lo que te molesta", "Respira hondo", "Busca algo positivo"],
    'neutral':      ["Todo esta en calma", "Buen momento para concentrarte", "Organiza tus ideas"],
}

# ── Estado global ──
resultado_actual = {
    'genero':          'N/A',
    'edad':            'N/A',
    'emocion':         'N/A',
    'confianza':       0.0,
    'recomendaciones': []
}
lock = threading.Lock()

# Para detectar persona nueva
persona_anterior  = {'genero': None, 'edad_rango': None}
nueva_persona_data = {'activo': False, 'genero': 'N/A', 'edad': 'N/A'}

# Suavizado de edad (media móvil de las últimas N lecturas)
historial_edad = []
VENTANA_EDAD   = 6   # frames para promediar


def obtener_ip_local():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('192.168.1.1', 1))
        return s.getsockname()[0]
    except Exception:
        return '127.0.0.1'
    finally:
        s.close()


def edad_a_rango(edad):
    """Convierte edad a rango para detectar cambio de persona."""
    if edad == 'N/A':
        return 'N/A'
    e = int(edad)
    if e < 13:   return '0-12'
    elif e < 18: return '13-17'
    elif e < 25: return '18-24'
    elif e < 35: return '25-34'
    elif e < 50: return '35-49'
    else:        return '50+'


def mapear_emocion(emocion_raw, genero):
    """Traduce emociones DeepFace al español con género correcto."""
    sufijo = 'o' if genero == 'Hombre' else 'a'
    mapa = {
        'happy':    'feliz',
        'sad':      'triste',
        'angry':    f'enoj{sufijo}',
        'fear':     'miedoso' if sufijo == 'o' else 'miedosa',
        'disgust':  f'disgust{sufijo}',
        'surprise': f'sorprendid{sufijo}',
        'neutral':  'neutral',
    }
    return mapa.get(emocion_raw, 'neutral')


# Umbral mínimo de confianza para aceptar una emoción (0–100 %)
CONFIANZA_MINIMA = 35.0

# Umbral combinado fear+angry para detectar estrés (0–100 %)
UMBRAL_ESTRES = 40.0


def analizar_frame(frame):
    global resultado_actual, persona_anterior, nueva_persona_data, historial_edad

    try:
        resultado = DeepFace.analyze(
            frame,
            actions=['emotion', 'age', 'gender'],
            enforce_detection=False,
            silent=True
        )

        if isinstance(resultado, list):
            resultado = resultado[0]

        # ── Género ──
        genero_raw = resultado.get('dominant_gender', '')
        if 'woman' in genero_raw.lower() or 'female' in genero_raw.lower():
            genero = 'Mujer'
        elif 'man' in genero_raw.lower() or 'male' in genero_raw.lower():
            genero = 'Hombre'
        else:
            genero = 'N/A'

        # ── Edad suavizada (media móvil) ──
        edad_raw = resultado.get('age', None)
        if edad_raw is not None:
            historial_edad.append(int(edad_raw))
            if len(historial_edad) > VENTANA_EDAD:
                historial_edad.pop(0)
            edad = round(sum(historial_edad) / len(historial_edad))
        else:
            edad = 'N/A'

        rango = edad_a_rango(edad)

        # ── Emoción con confianza mínima ──
        emociones   = resultado.get('emotion', {})
        emocion_raw = resultado.get('dominant_emotion', 'neutral')
        confianza   = emociones.get(emocion_raw, 0.0)

        if confianza < CONFIANZA_MINIMA:
            # Confianza baja: mantener última emoción conocida o neutral
            with lock:
                emocion    = resultado_actual.get('emocion', 'neutral') or 'neutral'
                confianza  = resultado_actual.get('confianza', 0.0)
        else:
            # Detectar estrés: miedo + enojo altos al mismo tiempo
            fear_pct  = emociones.get('fear',  0.0)
            angry_pct = emociones.get('angry', 0.0)
            if (fear_pct + angry_pct) > UMBRAL_ESTRES and emocion_raw in ('fear', 'angry'):
                emocion = 'estresado' if genero == 'Hombre' else 'estresada'
            else:
                emocion = mapear_emocion(emocion_raw, genero)

        # ── Detectar si es una persona nueva ──
        with lock:
            es_nueva = (
                persona_anterior['genero']     != genero or
                persona_anterior['edad_rango'] != rango
            )
            if es_nueva and genero != 'N/A':
                persona_anterior['genero']     = genero
                persona_anterior['edad_rango'] = rango
                nueva_persona_data = {
                    'activo': True,
                    'genero': genero,
                    'edad':   str(edad)
                }

            resultado_actual = {
                'genero':          genero,
                'edad':            edad,
                'emocion':         emocion,
                'confianza':       round(confianza, 1),
                'recomendaciones': recomendaciones.get(emocion, recomendaciones['neutral'])
            }

    except Exception as e:
        print(f"[Análisis] Error: {e}")

    return frame


def generar_frames():
    cap = cv2.VideoCapture(0)
    frame_count = 0

    if not cap.isOpened():
        frame_err = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame_err, "Conecta una camara", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        _, buf = cv2.imencode('.jpg', frame_err)
        fb = buf.tobytes()
        while True:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + fb + b'\r\n')
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        # Analizar cada 10 frames para no sobrecargar la CPU
        if frame_count % 10 == 0:
            threading.Thread(
                target=analizar_frame,
                args=(frame.copy(),),
                daemon=True
            ).start()

        # Dibujar resultados sobre el frame
        with lock:
            g  = resultado_actual['genero']
            e  = resultado_actual['edad']
            em = resultado_actual['emocion']
            cf = resultado_actual['confianza']

        cv2.putText(frame, f"Genero: {g}",           (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 180), 2)
        cv2.putText(frame, f"Edad:   {e}",            (10, 58),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 180), 2)
        cv2.putText(frame, f"Emocion: {em} ({cf}%)",  (10, 86),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 255), 2)

        _, buf = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video')
def video():
    return Response(generar_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/nueva-persona')
def nueva_persona_endpoint():
    global nueva_persona_data
    with lock:
        data = nueva_persona_data.copy()
        nueva_persona_data['activo'] = False   # reset tras leerlo
    return jsonify(data)


@app.route('/resultado')
def resultado():
    with lock:
        return jsonify(resultado_actual)


@app.route('/info')
def info():
    return jsonify({
        'sistema': platform.system(),
        'python':  platform.python_version(),
        'ip':      obtener_ip_local()
    })


if __name__ == '__main__':
    ip = obtener_ip_local()
    print("=" * 50)
    print("  ANALIZADOR DE EMOCIONES")
    print("=" * 50)
    print(f"  Local : http://localhost:5000")
    print(f"  Red   : http://{ip}:5000")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5000, debug=False)