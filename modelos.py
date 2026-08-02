"""
Ejecuta este script UNA sola vez para descargar los modelos a tu computadora.
Despues de esto, la pagina carga instantaneamente sin necesitar internet.

Como usarlo:
  1. Abre una terminal en esta carpeta
  2. Escribe:  python descargar_modelos.py
  3. Espera que termine
  4. Listo!
"""
import urllib.request
import os

# Carpeta donde esta este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEST     = os.path.join(BASE_DIR, "models")

print(f"Carpeta de destino: {DEST}")

if not os.path.exists(DEST):
    os.makedirs(DEST)
    print("Carpeta 'models' creada.")
else:
    print("Carpeta 'models' ya existe.")

# URL correcta: repositorio oficial de modelos en GitHub (raw)
BASE_URL = "https://raw.githubusercontent.com/justadudewhohacks/face-api.js/master/weights"

archivos = [
    "tiny_face_detector_model-weights_manifest.json",
    "tiny_face_detector_model-shard1",
    "face_expression_model-weights_manifest.json",
    "face_expression_model-shard1",
    "age_gender_model-weights_manifest.json",
    "age_gender_model-shard1",
]

print()
print("=" * 55)
print("  Descargando modelos de reconocimiento facial...")
print("=" * 55)

errores = 0
for nombre in archivos:
    destino = os.path.join(DEST, nombre)
    if os.path.exists(destino) and os.path.getsize(destino) > 500:
        print(f"  Ya existe: {nombre}")
        continue
    url = f"{BASE_URL}/{nombre}"
    print(f"  Descargando: {nombre} ...")
    try:
        urllib.request.urlretrieve(url, destino)
        kb = os.path.getsize(destino) // 1024
        print(f"  OK  ({kb} KB)")
    except Exception as e:
        print(f"  ERROR: {e}")
        errores += 1

print()
print("=" * 55)
if errores == 0:
    print("  Descarga completa! Todos los modelos estan listos.")
    print()
    print("  Ahora corre:  python -m http.server 5500")
    print("  Y abre:       http://localhost:5500")
else:
    print(f"  Hubo {errores} errores. Verifica tu conexion a internet")
    print("  y vuelve a ejecutar este script.")
print("=" * 55)
input("\nPresiona Enter para cerrar...")