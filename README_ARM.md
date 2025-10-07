# Optimización ARM/Raspberry Pi - Guía de Uso

## Descripción
Esta versión del detector de personas está optimizada específicamente para dispositivos ARM como Raspberry Pi, usando NCNN como motor de inferencia alternativo para mejor rendimiento.

## Mejoras de Rendimiento ARM

### 1. Motor de Inferencia NCNN
- **2-3x más rápido** que PyTorch en ARM
- Menor uso de memoria
- Optimizado para CPUs ARM con NEON
- Sin dependencia de GPU

### 2. Configuraciones Automáticas ARM
- Resolución reducida automáticamente (320x240)
- FPS limitados (10 FPS en lugar de 15)
- Frame skip activado por defecto (2 frames)
- Tamaño de modelo reducido (256px en lugar de 320px)

### 3. Monitoreo de Hardware
- Temperatura del CPU en tiempo real
- Advertencias de sobrecalentamiento
- Estadísticas de rendimiento específicas para ARM

## Instalación en Raspberry Pi

### 1. Dependencias del Sistema
```bash
# Actualizar sistema
sudo apt update && sudo apt upgrade -y

# Instalar dependencias OpenCV
sudo apt install -y python3-opencv libopencv-dev

# Instalar FFmpeg (opcional, para video output)
sudo apt install -y ffmpeg

# Habilitar cámara (si usas la Pi Camera)
sudo raspi-config  # Enable Camera
```

### 2. Dependencias Python
```bash
# Crear entorno virtual
python3 -m venv venv_yolo
source venv_yolo/bin/activate

# Instalar PyTorch para CPU ARM
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Instalar NCNN
pip install ncnn

# Instalar otras dependencias
pip install ultralytics pandas numpy opencv-python
```

### 3. Convertir Modelo a NCNN
```bash
# Convertir YOLOv8n a formato NCNN
python convert_yolo_to_ncnn.py --input yolov8n.pt --output yolov8n_arm --imgsz 256

# Esto generará:
# - yolov8n_arm.param (arquitectura del modelo)
# - yolov8n_arm.bin (pesos del modelo)
```

## Uso Optimizado para ARM

### Comando Básico (Automático)
```bash
# El script detecta ARM automáticamente y usa configuraciones optimizadas
python yolo-detection.py --source 0 --monitor-temp
```

### Comando con NCNN (Recomendado)
```bash
python yolo-detection.py \
    --source 0 \
    --use-ncnn \
    --ncnn-param yolov8n_arm.param \
    --ncnn-bin yolov8n_arm.bin \
    --monitor-temp
```

### Modo Ultra-Eficiente
```bash
python yolo-detection.py \
    --source 0 \
    --use-ncnn \
    --ncnn-param yolov8n_arm.param \
    --ncnn-bin yolov8n_arm.bin \
    --width 256 \
    --height 192 \
    --frame-skip 3 \
    --headless \
    --monitor-temp
```

## Parámetros Específicos ARM

### Nuevos Argumentos
- `--use-ncnn`: Forzar uso de NCNN (recomendado para ARM)
- `--ncnn-param`: Archivo .param del modelo NCNN
- `--ncnn-bin`: Archivo .bin del modelo NCNN
- `--monitor-temp`: Mostrar temperatura del CPU

### Configuraciones Automáticas ARM
Cuando se detecta Raspberry Pi, se aplican automáticamente:
- `--width 320` (en lugar de 640)
- `--height 240` (en lugar de 480)
- `--imgsz 256` (en lugar de 320)
- `--fps 10` (en lugar de 15)
- `--frame-skip 2` (en lugar de 0)

## Rendimiento Esperado

### Raspberry Pi 4 (4GB)
- **Con PyTorch**: ~3-5 FPS @ 320x240
- **Con NCNN**: ~8-12 FPS @ 320x240
- **Modo eficiente**: ~15-20 FPS @ 256x192

### Raspberry Pi 3B+
- **Con PyTorch**: ~1-2 FPS @ 320x240
- **Con NCNN**: ~4-6 FPS @ 320x240
- **Modo eficiente**: ~8-10 FPS @ 256x192

## Consejos de Optimización

### 1. Gestión Térmica
```bash
# Monitorear temperatura continuamente
python yolo-detection.py --source 0 --monitor-temp

# Si la temperatura supera los 70°C, considera:
# - Reducir la resolución
# - Aumentar frame-skip
# - Añadir ventilación
```

### 2. Optimización de Red
```bash
# Para cámaras IP o streaming remoto
python yolo-detection.py \
    --source "rtsp://ip:port/stream" \
    --use-ncnn \
    --ncnn-param yolov8n_arm.param \
    --ncnn-bin yolov8n_arm.bin \
    --frame-skip 4 \
    --headless
```

### 3. Grabación Eficiente
```bash
# Grabar video sin mostrar interfaz (menor CPU)
python yolo-detection.py \
    --source video.mp4 \
    --use-ncnn \
    --ncnn-param yolov8n_arm.param \
    --ncnn-bin yolov8n_arm.bin \
    --headless \
    --output-video resultado_arm.mp4
```

## Troubleshooting ARM

### Error: "NCNN no disponible"
```bash
pip install ncnn
# o compilar desde fuente para mejor optimización:
# git clone https://github.com/Tencent/ncnn.git
```

### Error: "Temperatura muy alta"
- Verifica ventilación
- Reduce resolución: `--width 256 --height 192`
- Aumenta frame skip: `--frame-skip 4`
- Usa modo headless: `--headless`

### Error: "FPS muy bajos"
- Usa NCNN: `--use-ncnn`
- Reduce resolución al mínimo
- Aumenta frame skip
- Considera usar modelo más pequeño (nano en lugar de small)

## Comparación de Rendimiento

| Configuración | Pi 4 FPS | Pi 3B+ FPS | Uso CPU | Temperatura |
|---------------|----------|------------|---------|-------------|
| PyTorch 640x480 | 2-3 | 1 | 95-100% | 75-80°C |
| PyTorch 320x240 | 4-5 | 2 | 85-90% | 70-75°C |
| NCNN 320x240 | 10-12 | 5-6 | 70-80% | 65-70°C |
| NCNN 256x192 | 15-18 | 8-10 | 60-70% | 60-65°C |

## Modelos Recomendados para ARM

1. **YOLOv8n** (Nano): Más rápido, precisión aceptable
2. **YOLOv8s** (Small): Balance velocidad/precisión
3. **Evitar YOLOv8m/l/x**: Demasiado pesados para ARM

## Archivo de Configuración de Ejemplo

Crea `config_arm.sh`:
```bash
#!/bin/bash
python yolo-detection.py \
    --source 0 \
    --use-ncnn \
    --ncnn-param models/yolov8n_arm.param \
    --ncnn-bin models/yolov8n_arm.bin \
    --width 320 \
    --height 240 \
    --frame-skip 2 \
    --monitor-temp \
    --draw-boxes \
    --conf 0.6
```

```bash
chmod +x config_arm.sh
./config_arm.sh
```