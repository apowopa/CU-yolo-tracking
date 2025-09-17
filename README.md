# Detección y Seguimiento de Personas con YOLO

## RECOMENDACIONES DE INSTALACIÓN DE ULTRALYTICS

Si tu computadora no tiene una GPU compatible con CUDA, utiliza el siguiente comando para instalar ultralytics:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision numpy ultralytics
```

## comando de pruebas
```bash
python yolo-detection.py --source viedo-prueba01.mp4 --target-fps 30 --frame-skip 2 --ground-truth prueba01.csv --save-events out-prueba01.csv --tracker botsort.yaml --output-video out-prueba01.mp4
```

## Descripción General
Implementación avanzada para detectar, seguir y contar personas con YOLOv8 y ByteTrack en videos o cámaras. El sistema permite contar personas que cruzan una línea definida, evaluar el rendimiento contra datos de referencia y generar salidas de video de alta calidad.

## Funcionalidades

- **Detección**: Modelos YOLOv8, filtrado por confianza, soporte multi-clase
- **Seguimiento**: ByteTrack/BotSORT con IDs persistentes y recuperación automática
- **Conteo de Personas**: Conteo automático de personas que cruzan una línea definible
- **Control de Reproducción**: Pausa/reanudación con espacio, definición de línea de conteo interactiva
- **Optimización**: 
  - Redimensionamiento automático de videos
  - Control de FPS y salto de frames
  - Soporte para half precision (FP16) en GPU
  - Modo headless para entornos sin interfaz gráfica
- **Evaluación**: Comparación contra ground truth con métricas MAE y error porcentual
- **Registro de Datos**: Exportación de eventos de cruce a CSV con pandas
- **Visualización**: 
  - Cuadros delimitadores con IDs
  - Contador de FPS en tiempo real
  - HUD con estadísticas y métricas
- **Exportación de Video**: 
  - Grabación de video con toda la interfaz visual
  - Conversión a MP4 de alta calidad usando FFmpeg
  - Parámetros configurables de calidad y FPS

## Uso

Uso básico con parámetros predeterminados:
```bash
python yolo-detection.py --source 0
```

Usando un archivo de video con parámetros personalizados:
```bash
python yolo-detection.py --source video.mp4 --conf 0.5 --width 640 --height 480
```

Conteo con evaluación y salida de video:
```bash
python yolo-detection.py --source video.mp4 --ground-truth datos.csv --output-video resultado
```

### Argumentos de Línea de Comandos

#### Entrada y Modelo
- `--source`: Índice de cámara o ruta de archivo de video (predeterminado: "0")
- `--model`: Modelo YOLOv8 a utilizar (predeterminado: "yolov8n.pt")
- `--conf`: Umbral mínimo de confianza (predeterminado: 0.5)
- `--classes`: Clases a detectar (predeterminado: [0] para persona)
- `--imgsz`: Tamaño de imagen para inferencia (predeterminado: 320)

#### Control de Visualización
- `--width`: Ancho máximo de fotograma (predeterminado: 640)
- `--height`: Altura máxima de fotograma (predeterminado: 480)
- `--draw-boxes`: Dibujar cuadros delimitadores (predeterminado: activado)
- `--headless`: Modo sin interfaz gráfica, solo salida por consola

#### Control de Rendimiento
- `--fps`: FPS máximos para captura de cámara (predeterminado: 15)
- `--target-fps`: FPS objetivo para reproducción de videos (0=velocidad máxima)
- `--frame-skip`: Saltar N frames por cada frame procesado (0=desactivado)
- `--half`: Usar half precision (FP16) para reducir consumo de memoria en GPU

#### Seguimiento
- `--tracker`: Tipo de tracker (predeterminado: "bytetrack.yaml", alternativa: "botsort.yaml")

#### Evaluación y Registro
- `--ground-truth`: Ruta al archivo CSV con ground truth de conteos [time,in,out]
- `--save-events`: Guardar eventos de cruce en archivo CSV [frame,track_id,event,count]

#### Salida de Video
- `--output-video`: Guardar video con UI en archivo MP4 usando FFmpeg (ej: output.mp4)

## Controles Interactivos

Durante la ejecución, los siguientes controles están disponibles:

- `Espacio`: Pausar/reanudar la reproducción del video
- `l`: Activar modo de definición de línea (requiere dos clics para definir los puntos)
- `+`: Aumentar el salto de frames
- `-`: Disminuir el salto de frames
- `Esc` o `q`: Salir del programa

## Requisitos
- Python 3.6+
- OpenCV
- Ultralytics YOLOv8
- NumPy
- Pandas
- FFmpeg (opcional, para conversión de video a MP4 de alta calidad)

## Instalación de FFmpeg

Para aprovechar la conversión de video de alta calidad, es recomendable instalar FFmpeg:

### En Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install ffmpeg
```

### En macOS (con Homebrew):
```bash
brew install ffmpeg
```

### En Windows:
Descargar de [FFmpeg.org](https://ffmpeg.org/download.html) y añadir al PATH.

## Formato de Archivos

### Ground Truth (CSV)
```
time,in,out
10,2,0
20,5,1
30,8,3
```

### Eventos de Salida (CSV)
```
frame,track_id,event,count_in,count_out,time_sec
120,1,IN,1,0,4.5
180,2,IN,2,0,6.2
240,1,OUT,2,1,8.3
```
