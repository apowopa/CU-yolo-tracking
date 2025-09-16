# Detección y Seguimiento de Personas con YOLO

# RECOMENDACIONES DE INSTALACION DE ULTRALYTICS

si tu computadora no tiene una gpu compatible con cuda, utiliza el siguiente comando para instalar ultralytics:

```
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision numpy ultralytics

```

## Descripción General
Este repositorio contiene una implementación en Python para la detección y seguimiento de personas utilizando modelos YOLOv8 y ByteTrack. El script proporciona detección y seguimiento eficiente de personas tanto en archivos de video como en cámaras, con características de optimización de rendimiento.

## Funcionalidades

### Características de Detección
- Detección de personas utilizando modelos YOLOv8
- Filtrado de detecciones basado en confianza
- Cambio automático a modo de solo detección cuando el seguimiento falla
- Soporte para múltiples clases de objetos (predeterminado: persona)

### Características de Seguimiento
- Seguimiento de objetos con el algoritmo ByteTrack
- IDs de seguimiento persistentes para las personas detectadas
- Implementación de seguimiento no bloqueante con manejo de tiempos de espera
- Recuperación automática de fallos de seguimiento

### Optimización de Rendimiento
- Redimensionamiento automático de fotogramas para videos de alta resolución
- Limitación de FPS para cámaras
- Control de resolución para reducir la carga computacional
- Seguimiento basado en hilos para evitar congelamiento de la interfaz

### Características de Visualización
- Visualización de cuadros delimitadores para personas detectadas
- Visualización de IDs de seguimiento con código de colores para cada persona
- Contador de FPS en tiempo real e información de resolución
- Indicador de modo (Seguimiento/Detección)

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
