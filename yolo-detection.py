import argparse
import cv2
import numpy as np
import sys
import time
import threading
from ultralytics import YOLO

def try_open_capture(device, width=640, height=480, fps=15, prefer_mjpg=True, use_v4l2=True):
    """Abre una fuente de video (cámara o archivo) y configura su resolución y FPS.
    
    Para cámaras: Intenta establecer la resolución y FPS directamente.
    Para archivos de video: Lee la resolución original y redimensiona si es necesario.
    
    Args:
        device: Índice de cámara (int) o ruta a archivo de video (str)
        width: Ancho máximo deseado
        height: Alto máximo deseado
        fps: FPS máximos deseados (solo afecta a cámaras)
        prefer_mjpg: Si se prefiere el códec MJPG para cámaras
        use_v4l2: Si se usa V4L2 para cámaras en Linux
        
    Returns:
        (cap, (eff_w, eff_h, eff_fps, eff_fourcc)): VideoCapture y dimensiones efectivas
    """
    # Determinar si es una cámara o un archivo de video
    is_camera = isinstance(device, int) or (isinstance(device, str) and device.isdigit())
    
    # Abrir la captura con las banderas adecuadas
    cap_flags = cv2.CAP_V4L2 if use_v4l2 and is_camera else 0
    cap = cv2.VideoCapture(device, cap_flags)
    if not cap.isOpened():
        return None, None
    
    # Obtener propiedades originales
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if is_camera:
        # Para cámaras, intentar establecer los parámetros directamente
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
        if prefer_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*'MJPG'))
    else:
        # Para archivos de video, verificamos si necesita redimensionamiento
        need_resize = (orig_w > width or orig_h > height) and width > 0 and height > 0
        
        if need_resize:
            # Calculamos la nueva resolución manteniendo la relación de aspecto
            aspect_ratio = orig_w / orig_h
            if orig_w > orig_h:
                new_w = min(width, orig_w)
                new_h = int(new_w / aspect_ratio)
                if new_h > height:
                    new_h = height
                    new_w = int(new_h * aspect_ratio)
            else:
                new_h = min(height, orig_h)
                new_w = int(new_h * aspect_ratio)
                if new_w > width:
                    new_w = width
                    new_h = int(new_w / aspect_ratio)
                    
            print(f"Redimensionando video de {orig_w}x{orig_h} a {new_w}x{new_h}")
            
            # Crear una clase para manejar el redimensionamiento automático
            class ResizeCapture:
                def __init__(self, cap, width, height):
                    self.cap = cap
                    self.width = width
                    self.height = height
                    
                def read(self):
                    ret, frame = self.cap.read()
                    if not ret:
                        return ret, frame
                    return ret, cv2.resize(frame, (self.width, self.height))
                    
                def isOpened(self):
                    return self.cap.isOpened()
                    
                def get(self, prop_id):
                    if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
                        return self.width
                    elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
                        return self.height
                    return self.cap.get(prop_id)
                    
                def set(self, prop_id, value):
                    return self.cap.set(prop_id, value)
                    
                def release(self):
                    return self.cap.release()
            
            # Reemplazar el VideoCapture original con nuestra versión que redimensiona
            cap = ResizeCapture(cap, new_w, new_h)
            
    # Obtener las dimensiones y FPS efectivos
    eff_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    eff_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    eff_fps = cap.get(cv2.CAP_PROP_FPS)
    eff_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    return cap, (eff_w, eff_h, eff_fps, eff_fourcc)


def main():
    ap = argparse.ArgumentParser(description="Tracking de personas con YOLOv8")
    ap.add_argument("--source", type=str, default="0", help="Ruta a video o índice de cámara")
    ap.add_argument("--model", type=str, default="yolov8n.pt", help="Modelo YOLOv8 a usar")
    ap.add_argument("--conf", type=float, default=0.4, help="Confianza mínima para detección")
    ap.add_argument("--classes", nargs='+', type=int, default=[0], help="Clases a detectar (0=persona)")
    ap.add_argument("--imgsz", type=int, default=320, help="Tamaño de imagen para inferencia")
    ap.add_argument("--width", type=int, default=640, help="Ancho máximo de procesamiento")
    ap.add_argument("--height", type=int, default=480, help="Alto máximo de procesamiento")
    ap.add_argument("--fps", type=int, default=15, help="FPS máximo para captura de cámara")
    ap.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tipo de tracker (bytetrack.yaml, botsort.yaml)")
    args = ap.parse_args()

    print(f"Cargando modelo {args.model}...")
    model = YOLO(args.model)
    
    # Verificar si el modelo soporta tracking
    if not hasattr(model, 'track'):
        print("[ERROR] El modelo no soporta tracking. Asegúrate de usar YOLOv8.")
        sys.exit(1)
        
    print(f"Usando tracker: {args.tracker}")

    source_input = int(args.source) if args.source.isdigit() else args.source
    
    # Abrir el video o cámara con la función mejorada que maneja redimensionamiento
    cap, eff = try_open_capture(source_input, args.width, args.height, args.fps)
    if cap is None:
        print(f"[ERROR] No se pudo abrir la fuente: '{args.source}'")
        sys.exit(1)

    eff_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    eff_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    eff_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Captura iniciada: {eff_w}x{eff_h} @ {eff_fps:.1f}FPS")
    win = "Tracking con YOLOv8"
    cv2.namedWindow(win)
    
    # Variable para medir FPS
    last_time = time.time()
    fps_counter = 0
    fps_display = 0
    tracking_timeout = 0.5  # Timeout en segundos para tracking
    
    # Variable para controlar si estamos usando tracking o detección simple
    using_tracking = True

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crear una copia del frame para dibujar
        vis_frame = frame.copy()
        
        # Medir tiempo para FPS
        current_time = time.time()
        elapsed = current_time - last_time
        fps_counter += 1
        
        # Actualizar contador de FPS cada segundo
        if elapsed >= 1.0:
            fps_display = fps_counter / elapsed
            fps_counter = 0
            last_time = current_time
            
        if using_tracking:
            class TrackingThread(threading.Thread):
                def __init__(self, model, frame, args):
                    threading.Thread.__init__(self)
                    self.model = model
                    self.frame = frame
                    self.args = args
                    self.results = None
                    
                def run(self):
                    try:
                        self.results = self.model.track(
                            self.frame, 
                            conf=self.args.conf, 
                            classes=self.args.classes, 
                            imgsz=self.args.imgsz, 
                            tracker=self.args.tracker,
                            persist=True,
                            verbose=False
                        )
                    except Exception as e:
                        print(f"Error en tracking: {e}")
                        self.results = None
            
            tracking_thread = TrackingThread(model, frame, args)
            tracking_thread.start()
            tracking_thread.join(timeout=tracking_timeout)
            
            # Si el hilo sigue vivo después del timeout, matarlo y usar predict
            if tracking_thread.is_alive():
                print("Tracking timeout - cambiando a predict")
                using_tracking = False
                results = model.predict(frame, conf=args.conf, classes=args.classes, imgsz=args.imgsz, verbose=False)
            else:
                # Si el tracking funcionó, usar los resultados
                if tracking_thread.results is not None:
                    results = tracking_thread.results
                else:
                    # Si hubo un error en el tracking, usar predict
                    print("Error en tracking - cambiando a predict")
                    using_tracking = False
                    results = model.predict(frame, conf=args.conf, classes=args.classes, imgsz=args.imgsz, verbose=False)
        else:
            # Si estamos en modo predict, intentar volver a tracking cada 30 frames
            if fps_counter % 30 == 0:
                using_tracking = True
                print("Intentando volver a tracking")
            results = model.predict(frame, conf=args.conf, classes=args.classes, imgsz=args.imgsz, verbose=False)
        
        # Mostrar FPS, resolución y modo
        mode_text = "Tracking" if using_tracking else "Detección"
        cv2.putText(vis_frame, f"FPS: {fps_display:.1f} | Res: {eff_w}x{eff_h} | Modo: {mode_text}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Obtener los resultados del tracking            
        if results[0].boxes is not None:
            # Convertir tensores a numpy arrays
            boxes = results[0].boxes.xyxy
            if hasattr(boxes, 'cpu'):  # Si es un tensor
                boxes = boxes.cpu().numpy()
            boxes = boxes.astype(int)
            
            confs = results[0].boxes.conf
            if hasattr(confs, 'cpu'):  # Si es un tensor
                confs = confs.cpu().numpy()
            
            # Verificar si tenemos track_ids
            has_track_ids = False
            track_ids = None
            if hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id
                if hasattr(track_ids, 'cpu'):
                    track_ids = track_ids.cpu().numpy()
                if len(track_ids) > 0:
                    track_ids = track_ids.astype(int)
                    has_track_ids = True
            
            # Dibujar cada detección con su ID
            for i, box in enumerate(boxes):
                try:
                    x1, y1, x2, y2 = box
                    conf = confs[i]
                    
                    # Determinar color y etiqueta según si tenemos track_id
                    if has_track_ids and track_ids is not None:
                        track_id = track_ids[i]
                        # Color basado en el ID para diferenciar visualmente
                        color_id = int((track_id * 50) % 255)
                        color = (int(color_id), int(255 - color_id), 150)
                        label = f"ID:{track_id} {conf:.2f}"
                    else:
                        # Color por defecto para detecciones sin ID
                        color = (0, 255, 0)
                        label = f"Conf: {conf:.2f}"
                    
                    # Dibujar bounding box
                    cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Dibujar ID y confianza
                    cv2.putText(vis_frame, label, (int(x1), int(y1 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(vis_frame, label, (int(x1), int(y1 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                except Exception as e:
                    print(f"Error procesando detección {i}: {e}")

        cv2.imshow(win, vis_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()