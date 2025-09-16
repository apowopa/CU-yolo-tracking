import argparse
import cv2
import numpy as np
import sys
import time
from ultralytics import YOLO

def try_open_capture(device, width=640, height=640, fps=15, prefer_mjpg=True, use_v4l2=True):
    cap_flags = cv2.CAP_V4L2 if use_v4l2 and (isinstance(device, int) or str(device).startswith('/dev/')) else 0
    cap = cv2.VideoCapture(device, cap_flags)
    if not cap.isOpened():
        return None, None
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps:
        cap.set(cv2.CAP_PROP_FPS, fps)
    if prefer_mjpg and (isinstance(device, int) or str(device).startswith('/dev/')):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
    eff_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    eff_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    eff_fps = cap.get(cv2.CAP_PROP_FPS)
    eff_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    return cap, (eff_w, eff_h, eff_fps, eff_fourcc)


def main():
    ap = argparse.ArgumentParser(description="Detección de personas con YOLOv8")
    ap.add_argument("--source", type=str, default="0", help="Ruta a video o índice de cámara")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="Modelo YOLOv8 a usar")
    ap.add_argument("--conf", type=float, default=0.4, help="Confianza mínima para detección")
    ap.add_argument("--classes", nargs='+', type=int, default=[0], help="Clases a detectar (0=persona)")
    ap.add_argument("--imgsz", type=int, default=320, help="Tamaño de imagen para inferencia")
    ap.add_argument("--width", type=int, default=640, help="Ancho de captura")
    ap.add_argument("--height", type=int, default=480, help="Alto de captura")
    ap.add_argument("--fps", type=int, default=15, help="FPS de captura")
    args = ap.parse_args()

    print(f"Cargando modelo {args.model}...")
    model = YOLO(args.model)

    source_input = int(args.source) if args.source.isdigit() else args.source
    cap, eff = try_open_capture(source_input, args.width, args.height, args.fps)
    if cap is None:
        print(f"[ERROR] No se pudo abrir la fuente: '{args.source}'")
        sys.exit(1)

    eff_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    eff_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    eff_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Captura iniciada: {eff_w}x{eff_h} @ {eff_fps:.1f}FPS")

    win = "Detección con YOLOv8"
    cv2.namedWindow(win)
    
    # Variable para medir FPS
    last_time = time.time()
    fps_counter = 0
    fps_display = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Medir FPS
        current_time = time.time()
        elapsed = current_time - last_time
        fps_counter += 1
        
        # Actualizar FPS cada segundo
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            last_time = current_time
            
        # Procesar con YOLO directamente el frame
        results = model.predict(frame, conf=args.conf, classes=args.classes, imgsz=args.imgsz, verbose=False)
        
        # Mostrar FPS y resolución
        cv2.putText(frame, f"FPS: {fps_display:.1f} | Res: {eff_w}x{eff_h}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
        # Dibujar cajas de detección
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()