import argparse
import cv2
import numpy as np
import pandas as pd
import sys
import time
import torch
from ultralytics import YOLO
from evaluation import read_ground_truth, evaluate_counting


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
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    
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
                    
            print(f"Redimensionando video de {orig_w}x{orig_h}@{cap_fps:.1f}fps a {new_w}x{new_h}")
            
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
    ap.add_argument("--conf", type=float, default=0.5, help="Confianza mínima para detección")
    ap.add_argument("--classes", nargs='+', type=int, default=[0], help="Clases a detectar (0=persona)")
    ap.add_argument("--imgsz", type=int, default=320, help="Tamaño de imagen para inferencia")
    ap.add_argument("--width", type=int, default=640, help="Ancho máximo de procesamiento")
    ap.add_argument("--height", type=int, default=480, help="Alto máximo de procesamiento")
    ap.add_argument("--fps", type=int, default=15, help="FPS máximo para captura de cámara")
    ap.add_argument("--target-fps", type=int, default=0, help="FPS objetivo para videos (0=velocidad máxima)")
    ap.add_argument("--frame-skip", type=int, default=0, help="Saltar N frames por cada frame procesado (0=desactivado)")
    ap.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tipo de tracker (bytetrack.yaml, botsort.yaml)")
    ap.add_argument("--half", action="store_true", help="Usar half precision (FP16) para reducir consumo de memoria")
    ap.add_argument("--headless", action="store_true", help="Modo sin interfaz gráfica (solo salida por consola)")
    ap.add_argument("--draw-boxes", action="store_true", default=True, help="Dibujar bounding boxes (desactivar para ahorrar recursos)")
    ap.add_argument("--ground-truth", type=str, default="", help="Ruta al archivo CSV con ground truth de conteos [time,in,out]")
    ap.add_argument("--save-events", type=str, default="", help="Guardar eventos de cruce en archivo CSV [frame,track_id,event,count]")
    ap.add_argument("--output-video", type=str, default="", help="Guardar video con UI en archivo de salida MP4 (ej: output.mp4)")
    args = ap.parse_args()

    print(f"Cargando modelo {args.model}...")
    model = YOLO(args.model)
    
    # Aplicar half precision si se solicita
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
        if args.half:
            print("Usando half precision (FP16) para reducir consumo de memoria")
            # Half precision solo funciona en GPU
            model.to(device).half()
    else:
        print("CUDA no disponible, usando CPU")
        model.to(device)
    
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
    
    # Registrar el tiempo de inicio para mostrar el tiempo transcurrido
    start_time = time.time()
    
    # Variable para el tiempo efectivo de reproducción
    effective_playback_time = 0
    last_frame_time = time.time()
    
    # Cargar ground truth si se especificó
    ground_truth = []
    if args.ground_truth:
        print(f"Cargando ground truth desde {args.ground_truth}...")
        ground_truth = read_ground_truth(args.ground_truth)
        print(f"Se cargaron {len(ground_truth)} puntos de ground truth")
    
    # Inicializar registro de eventos si se especificó
    events_df = None
    if args.save_events:
        try:
            # Crear DataFrame para almacenar eventos
            events_df = pd.DataFrame(columns=[
                'frame', 'track_id', 'event', 'count_in', 'count_out', 'time_sec'
            ])
            print(f"Eventos serán guardados en {args.save_events}")
        except Exception as e:
            print(f"Error al inicializar DataFrame para eventos: {e}")
            events_df = None
    
    # Inicializar VideoWriter si se especificó archivo de salida
    video_writer = None
    output_filename = ""  # Variable para guardar el nombre del archivo de salida
    
    if args.output_video:
        try:
            # Asegurar que la extensión del archivo sea .mp4
            output_filename = args.output_video
            if not output_filename.lower().endswith('.mp4'):
                output_filename += '.mp4'
                print(f"Añadiendo extensión .mp4 al archivo de salida: {output_filename}")
            
            # Definir codec de video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec MP4
            
            # Determinar los FPS a usar para el video de salida
            output_fps = 30  # FPS por defecto
            if args.target_fps > 0:  # Si se especificó target-fps, usarlo
                output_fps = args.target_fps / args.frame_skip
                print(f"Usando {output_fps} FPS para el video de salida (desde target-fps)")
            elif eff_fps > 0:  # Si no, usar los FPS del video original si están disponibles
                output_fps = eff_fps
                print(f"Usando {output_fps} FPS para el video de salida (desde video original)")
            else:
                print(f"Usando {output_fps} FPS para el video de salida (valor por defecto)")
                
            # Crear el VideoWriter (nombre archivo, codec, fps, resolución)
            video_writer = cv2.VideoWriter(
                output_filename, 
                fourcc, 
                output_fps,  # Usar los FPS determinados
                (eff_w, eff_h)  # Misma resolución que el video original
            )
            print(f"Se guardará el video con UI en: {output_filename}")
        except Exception as e:
            print(f"Error al inicializar VideoWriter: {e}")
            video_writer = None
    
    # Configuración de ventana solo si no estamos en modo headless
    win = "Tracking con YOLOv8"
    if not args.headless:
        cv2.namedWindow(win)
    
    # Variable para medir FPS
    last_time = time.time()
    fps_counter = 0
    fps_display = 0
    
    # Variables para el conteo de personas
    count_in = 0
    count_out = 0
    
    # Inicializar la línea de conteo (por defecto en la mitad horizontal)
    line_start = (eff_w // 2, 0)
    line_end = (eff_w // 2, eff_h)
    
    # Estado para definir la línea de conteo mediante clics
    define_line = False
    line_points = []
    
    # Variable para controlar la pausa del video
    paused = False
    
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
    
    # Registrar la función de callback solo si no estamos en modo headless
    if not args.headless:
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

    # Contadores de fotogramas perdidos y procesados para estadísticas
    processed_frames = 0
    total_frames = 0

    while True:
        # Control de framerate para videos
        if not is_camera and args.target_fps > 0:
            process_start = time.time()
        
        # Solo leemos un nuevo frame si no estamos en pausa o si es una cámara (que no se puede pausar)
        if not paused or is_camera:
            # Leer el frame actual
            ret, frame = cap.read()
            if not ret:
                break
                
            total_frames += 1
            
            # Actualizar el tiempo efectivo de reproducción solo si no está pausado
            current_time = time.time()
            if not paused:
                effective_playback_time += current_time - last_frame_time
            last_frame_time = current_time
            
        # Saltar frames si es necesario (solo para videos)
        if not is_camera and frame_skip > 0:
            frame_count += 1
            if frame_count % (frame_skip + 1) != 0:
                continue
        
        processed_frames += 1
        
        # Crear una copia del frame para dibujar solo si es necesario
        vis_frame = None
        if not args.headless or video_writer is not None:
            if frame is not None:
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
            
        # Ejecutar el tracking directamente sin threading
        try:
            # Al usar stream=True, results es un generador que debemos iterar
            results_generator = model.track(
                frame, 
                conf=args.conf, 
                classes=args.classes, 
                imgsz=args.imgsz, 
                tracker=args.tracker,
                persist=True,
                verbose=False,
                stream=True
            )
            # Obtener el primer (y único) resultado del generador
            r = next(results_generator)
        except Exception as e:
            print(f"Error en tracking: {e}")
            continue
        
        # Solo procesamos la visualización si no estamos en modo headless
        if not args.headless:
            # Calcular tiempo transcurrido
            elapsed_time = time.time() - start_time
            
            # Mostrar FPS, resolución y modo
            cv2.putText(vis_frame, f"FPS: {fps_display:.1f} | Res: {eff_w}x{eff_h} | Proc: {processed_frames}/{total_frames}", 
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
            
            # Mostrar tiempo transcurrido del video (solo tiempo efectivo de reproducción)
            cv2.putText(vis_frame, f"Tiempo: {int(elapsed_time)}s", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
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
        if r.boxes is not None:
            # Convertir tensores a numpy arrays
            boxes = r.boxes.xyxy
            if hasattr(boxes, 'cpu'):  # Si es un tensor
                boxes = boxes.cpu().numpy()
            boxes = boxes.astype(int)
            
            confs = r.boxes.conf
            if hasattr(confs, 'cpu'):  # Si es un tensor
                confs = confs.cpu().numpy()
            
            # Verificar si tenemos track_ids
            has_track_ids = False
            track_ids = None
            if hasattr(r.boxes, 'id') and r.boxes.id is not None:
                track_ids = r.boxes.id
                if hasattr(track_ids, 'cpu'):
                    track_ids = track_ids.cpu().numpy()
                if len(track_ids) > 0:
                    track_ids = track_ids.astype(int)
                    has_track_ids = True
            
            # Procesar cada detección
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
                                    
                                    # Registrar evento en DataFrame si está habilitado
                                    if events_df is not None:
                                        events_df = pd.concat([events_df, pd.DataFrame([{
                                            'frame': total_frames, 
                                            'track_id': track_id, 
                                            'event': 'IN', 
                                            'count_in': count_in, 
                                            'count_out': count_out, 
                                            'time_sec': effective_playback_time
                                        }])], ignore_index=True)
                                        
                                elif info['last_side'] < 0 and not info['counted_out']:
                                    count_out += 1
                                    info['counted_out'] = True
                                    print(f"Persona ID:{track_id} salió. Total OUT: {count_out}")
                                    
                                    # Registrar evento en DataFrame si está habilitado
                                    if events_df is not None:
                                        events_df = pd.concat([events_df, pd.DataFrame([{
                                            'frame': total_frames, 
                                            'track_id': track_id, 
                                            'event': 'OUT', 
                                            'count_in': count_in, 
                                            'count_out': count_out, 
                                            'time_sec': effective_playback_time
                                        }])], ignore_index=True)
                        
                        # Dibujar elementos visuales solo si no estamos en modo headless
                        if not args.headless:
                            # Dibujar punto en el centro con color según el lado
                            center_color = (0, 255, 0) if current_side > 0 else (0, 0, 255)
                            cv2.circle(vis_frame, (center_x, center_y), 4, center_color, -1)
                            
                            # Dibujar bounding box si está habilitado
                            if args.draw_boxes:
                                cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                                
                                # Dibujar ID y confianza
                                cv2.putText(vis_frame, label, (int(x1), int(y1 - 10)), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
                                cv2.putText(vis_frame, label, (int(x1), int(y1 - 10)), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                    else:
                        # Detecciones sin ID, solo dibujar si no estamos en modo headless
                        if not args.headless and args.draw_boxes:
                            # Color por defecto para detecciones sin ID
                            color = (0, 255, 0)
                            label = f"Conf: {conf:.2f}"
                            
                            # Dibujar bounding box
                            cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            
                            # Dibujar confianza
                            cv2.putText(vis_frame, label, (int(x1), int(y1 - 10)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 4, cv2.LINE_AA)
                            cv2.putText(vis_frame, label, (int(x1), int(y1 - 10)), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
                except Exception as e:
                    print(f"Error procesando detección {i}: {e}")

        # Visualización solo si no estamos en modo headless
        if not args.headless and vis_frame is not None:
            # Usamos el tiempo efectivo de reproducción en lugar del tiempo real transcurrido
            elapsed_time = effective_playback_time
            
            # Mostrar FPS, resolución y estadísticas de procesamiento
            cv2.putText(vis_frame, f"FPS: {fps_display:.1f} | Res: {eff_w}x{eff_h} | Proc: {processed_frames}/{total_frames}", 
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
            
            # Mostrar tiempo transcurrido
            cv2.putText(vis_frame, f"Tiempo: {int(elapsed_time)}s", 
                       (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
            # Mostrar estado de pausa si está pausado
            if paused and not is_camera:
                cv2.putText(vis_frame, "PAUSADO", 
                           (vis_frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Mostrar métricas de evaluación si hay ground truth
            if ground_truth:
                metrics = evaluate_counting(ground_truth, elapsed_time, count_in, count_out)
                if metrics:
                    # Primera línea: Valores actuales vs ground truth
                    eval_text1 = f"GT: IN={metrics['gt_in']} OUT={metrics['gt_out']} | Actual: IN={count_in} OUT={count_out}"
                    cv2.putText(vis_frame, eval_text1, 
                               (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    
                    # Segunda línea: MAE y error porcentual
                    eval_text2 = f"MAE: {metrics['mae_total']} | Error: {metrics['error_pct_total']:.1f}%"
                    cv2.putText(vis_frame, eval_text2, 
                               (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Mostrar instrucciones sobre cómo definir la línea
            if define_line:
                cv2.putText(vis_frame, "Definiendo linea: haz clic para punto 1, luego punto 2", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                # Si ya tenemos un punto, dibujarlo
                if len(line_points) == 1:
                    cv2.circle(vis_frame, line_points[0], 5, (0, 0, 255), -1)
            
            # Mostrar contadores
            cv2.putText(vis_frame, f"IN: {count_in} | OUT: {count_out} | TOTAL: {count_in + count_out}", 
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                       
            # Mostrar la imagen
            cv2.imshow(win, vis_frame)
            
            # Guardar frame en el video de salida si está habilitado
            if video_writer is not None and vis_frame is not None:
                video_writer.write(vis_frame)
            
        # Siempre mostrar contadores en modo headless
        else:
            if fps_counter % 30 == 0:  # Actualizar cada 30 frames para no saturar la consola
                print(f"FPS: {fps_display:.1f} | IN: {count_in} | OUT: {count_out} | TOTAL: {count_in + count_out}")
            
            # En modo headless, si queremos guardar video, necesitamos usar el vis_frame ya creado
            if video_writer is not None and vis_frame is not None:
                # Agregar información básica al frame
                # Dibujar la línea de conteo
                cv2.line(vis_frame, line_start, line_end, (0, 0, 255), 2)
                
                # Mostrar contadores
                cv2.putText(vis_frame, f"IN: {count_in} | OUT: {count_out} | TOTAL: {count_in + count_out}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0,255), 2)
                
                # Mostrar FPS y tiempo
                cv2.putText(vis_frame, f"FPS: {fps_display:.1f} | Tiempo: {int(effective_playback_time)}s", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Guardar el frame
                video_writer.write(vis_frame)
        
        # Control de framerate para videos
        if not is_camera and args.target_fps > 0:
            process_time = time.time() - process_start
            sleep_time = max(0, frame_time - process_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Control de teclado solo si no estamos en modo headless
        if not args.headless:
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            elif key == ord(' '):  # Tecla espacio para pausar/reanudar
                paused = not paused
                if paused:
                    print("Video pausado. Presiona espacio para continuar.")
                else:
                    print("Video reanudado.")
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
    
    # Liberar VideoWriter si está inicializado
    if video_writer is not None:
        video_writer.release()
        print(f"Video con UI guardado en: {output_filename}")
    
    if not args.headless:
        cv2.destroyAllWindows()
        
    # Guardar DataFrame de eventos si está habilitado
    if events_df is not None and args.save_events:
        try:
            # Guardar en CSV usando pandas
            events_df.to_csv(args.save_events, index=False)
            print(f"Eventos guardados en {args.save_events}")
        except Exception as e:
            print(f"Error al guardar eventos: {e}")
        
    # Mostrar estadísticas finales
    print("\nEstadísticas finales:")
    print(f"Frames procesados: {processed_frames}/{total_frames} ({processed_frames/total_frames*100:.1f}%)")
    print(f"Conteo de personas: IN: {count_in} | OUT: {count_out} | TOTAL: {count_in + count_out}")
    
    # Mostrar métricas de evaluación final si hay ground truth
    if ground_truth:
        final_metrics = evaluate_counting(ground_truth, effective_playback_time, count_in, count_out)
        if final_metrics:
            print("\nEvaluación con Ground Truth:")
            print(f"Ground Truth: IN={final_metrics['gt_in']} OUT={final_metrics['gt_out']} TOTAL={final_metrics['gt_in'] + final_metrics['gt_out']}")
            print(f"MAE: IN={final_metrics['mae_in']} OUT={final_metrics['mae_out']} TOTAL={final_metrics['mae_total']}")
            print(f"Error %: IN={final_metrics['error_pct_in']:.2f}% OUT={final_metrics['error_pct_out']:.2f}% TOTAL={final_metrics['error_pct_total']:.2f}%")

if __name__ == "__main__":
    main()