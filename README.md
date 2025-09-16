# Detección y Seguimiento de Personas con YOLO

# RECOMENDACIONES DE INSTALACION DE ULTRALYTICS

si tu computadora no tiene una gpu compatible con cuda, utiliza el siguiente comando para instalar ultralytics:

```
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision numpy ultralytics

```

## Descripción General
Implementación para detectar y seguir personas con YOLOv8 y ByteTrack en videos o cámaras.

## Funcionalidades

- **Detección**: Modelos YOLOv8, filtrado por confianza, soporte multi-clase
- **Seguimiento**: ByteTrack con IDs persistentes y recuperación automática
- **Optimización**: Redimensionamiento automático, limitación de FPS
- **Visualización**: Cuadros delimitadores con IDs, contador de FPS en tiempo real

## Uso

Uso básico con parámetros predeterminados:
```
python yolo-detection.py --source 0
```

Usando un archivo de video con parámetros personalizados:
```
python yolo-detection.py --source video.mp4 --conf 0.5 --width 640 --height 480
```

### Argumentos de Línea de Comandos

- `--source`: Índice de cámara o ruta de archivo de video (predeterminado: "0")
- `--model`: Modelo YOLOv8 a utilizar (predeterminado: "yolov8n.pt")
- `--conf`: Umbral mínimo de confianza (predeterminado: 0.4)
- `--classes`: Clases a detectar (predeterminado: [0] para persona)
- `--imgsz`: Tamaño de imagen para inferencia (predeterminado: 320)
- `--width`: Ancho máximo de fotograma (predeterminado: 640)
- `--height`: Altura máxima de fotograma (predeterminado: 480)
- `--fps`: FPS máximos para captura de cámara (predeterminado: 15)
- `--tracker`: Tipo de tracker (predeterminado: "bytetrack.yaml")

## Requisitos
- Python 3.6+
- OpenCV
- Ultralytics YOLOv8
- NumPy
