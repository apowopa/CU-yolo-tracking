import argparse
import cv2
import numpy as np
import sys
import time
import threading
from ultralytics import YOLO
from collections import defaultdict


def sign_of_line(A, B, P):
    """Determina de qué lado de una línea se encuentra un punto.
       1 si P está por encima/izquierda, -1 si está por debajo/derecha, 0 si está en la línea
    """
    return np.sign((B[0] - A[0]) * (P[1] - A[1]) - (B[1] - A[1]) * (P[0] - A[0]))


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
    ap.add_argument("--target-fps", type=int, default=0, help="FPS objetivo para videos (0=velocidad máxima)")
    ap.add_argument("--frame-skip", type=int, default=0, help="Saltar N frames por cada frame procesado (0=desactivado)")
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
    
    # Variables para el conteo de personas
    count_in = 0
    count_out = 0
    
    # Inicializar la línea de conteo (por defecto en la mitad horizontal)
    line_start = (0, eff_h // 2)
    line_end = (eff_w, eff_h // 2)
    
    # Estado para definir la línea de conteo mediante clics
    define_line = False
    line_points = []
    
    # Diccionario para almacenar información de los tracks
    tracks_info = {}
    
    # Función para manejar eventos de clic del mouse
    def mouse_callback(event, x, y, flags, param):
        nonlocal define_line, line_points, line_start, line_end
        
        if define_line and event == cv2.EVENT_LBUTTONDOWN:
            line_points.append((x, y))
            if len(line_points) == 2:
                line_start = line_points[0]
                line_end = line_points[1]
                define_line = False
                line_points = []
                print(f"Nueva línea de conteo definida: {line_start} a {line_end}")
    
    # Registrar la función de callback
    cv2.setMouseCallback(win, mouse_callback)
    
    # Determinar si estamos usando una cámara o un archivo de video
    is_camera = isinstance(source_input, int) or (isinstance(source_input, str) and source_input.isdigit())
    
    # Configuración para el control de framerate
    frame_time = 0  # Tiempo que debería tomar procesar un frame
    if not is_camera and args.target_fps > 0:
        frame_time = 1.0 / args.target_fps
    
    # Configurar el salto de frames para videos
    frame_skip = max(0, args.frame_skip)
    frame_count = 0
    process_start = time.time()  # Inicializar para evitar errores

    while True:
        # Control de framerate para videos
        if not is_camera and args.target_fps > 0:
            process_start = time.time()
            
        # Leer el frame actual
        ret, frame = cap.read()
        if not ret:
            break
            
        # Saltar frames si es necesario (solo para videos)
        if not is_camera and frame_skip > 0:
            frame_count += 1
            if frame_count % (frame_skip + 1) != 0:
                continue
        
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
        
        # Mostrar información de control de frames si no es cámara
        if not is_camera:
            frame_ctrl_text = f"Skip: {frame_skip}"
            if args.target_fps > 0:
                frame_ctrl_text += f" | Target FPS: {args.target_fps}"
            cv2.putText(vis_frame, frame_ctrl_text, 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Dibujar la línea de conteo
        cv2.line(vis_frame, line_start, line_end, (0, 0, 255), 2)
        
        # Mostrar instrucciones sobre cómo definir la línea
        if define_line:
            cv2.putText(vis_frame, "Definiendo linea: haz clic para punto 1, luego punto 2", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # Si ya tenemos un punto, dibujarlo
            if len(line_points) == 1:
                cv2.circle(vis_frame, line_points[0], 5, (0, 0, 255), -1)
        
        # Mostrar contadores
        cv2.putText(vis_frame, f"IN: {count_in} | OUT: {count_out} | TOTAL: {count_in + count_out}", 
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
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
                    
                    # Calcular el centro de la persona para verificar el cruce de línea
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Determinar color y etiqueta según si tenemos track_id
                    if has_track_ids and track_ids is not None:
                        track_id = track_ids[i]
                        # Color basado en el ID para diferenciar visualmente
                        color_id = int((track_id * 50) % 255)
                        color = (int(color_id), int(255 - color_id), 150)
                        label = f"ID:{track_id} {conf:.2f}"
                        
                        # Verificar en qué lado de la línea está la persona
                        current_side = sign_of_line(line_start, line_end, (center_x, center_y))
                        
                        # Actualizar la información del track
                        if track_id not in tracks_info:
                            tracks_info[track_id] = {
                                'prev_side': current_side,
                                'last_side': current_side,
                                'age': 1,
                                'counted_in': False,
                                'counted_out': False
                            }
                        else:
                            # Actualizar información
                            info = tracks_info[track_id]
                            info['prev_side'] = info['last_side']
                            info['last_side'] = current_side
                            info['age'] += 1
                            
                            # Verificar si ha cruzado la línea (con histéresis)
                            if info['age'] >= 3 and info['prev_side'] != info['last_side']:
                                if info['last_side'] > 0 and not info['counted_in']:
                                    count_in += 1
                                    info['counted_in'] = True
                                    print(f"Persona ID:{track_id} entró. Total IN: {count_in}")
                                elif info['last_side'] < 0 and not info['counted_out']:
                                    count_out += 1
                                    info['counted_out'] = True
                                    print(f"Persona ID:{track_id} salió. Total OUT: {count_out}")
                        
                        # Dibujar punto en el centro con color según el lado
                        center_color = (0, 255, 0) if current_side > 0 else (0, 0, 255)
                        cv2.circle(vis_frame, (center_x, center_y), 4, center_color, -1)
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
        
        # Control de framerate para videos
        if not is_camera and args.target_fps > 0:
            process_time = time.time() - process_start
            sleep_time = max(0, frame_time - process_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('l'):
            # Activar modo de definición de línea
            define_line = True
            line_points = []
            print("Modo de definición de línea activado. Haz dos clics para definir la línea.")
        elif key == ord('+') and not is_camera and frame_skip < 10:
            # Aumentar el salto de frames
            frame_skip += 1
            print(f"Salto de frames: {frame_skip}")
        elif key == ord('-') and not is_camera and frame_skip > 0:
            # Disminuir el salto de frames
            frame_skip -= 1
            print(f"Salto de frames: {frame_skip}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()