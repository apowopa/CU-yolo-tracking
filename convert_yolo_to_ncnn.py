#!/usr/bin/env python3
"""
Script para convertir modelos YOLOv8 a formato NCNN optimizado para ARM/Raspberry Pi

Uso:
    python convert_yolo_to_ncnn.py --input yolov8n.pt --output yolov8n_ncnn

Requisitos:
    - ultralytics
    - ncnn (opcional, para validación)
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics no está instalado. Instalar con: pip install ultralytics")
    sys.exit(1)


def convert_yolo_to_ncnn(input_path, output_path, imgsz=320):
    """
    Convierte un modelo YOLOv8 a formato NCNN
    
    Args:
        input_path (str): Ruta al modelo YOLOv8 (.pt)
        output_path (str): Nombre base para los archivos de salida
        imgsz (int): Tamaño de imagen para la conversión
    """
    
    print(f"Cargando modelo YOLOv8: {input_path}")
    try:
        model = YOLO(input_path)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return False
    
    print(f"Convirtiendo a NCNN con tamaño de imagen: {imgsz}")
    try:
        # Exportar a formato NCNN
        success = model.export(
            format='ncnn',
            imgsz=imgsz,
            optimize=False,
            int8=False,  # Cambiar a True para quantización int8 (menor precisión, mayor velocidad)
            dynamic=False,
            simplify=True
        )
        
        if success:
            # Los archivos se generan automáticamente con el sufijo _ncnn_model
            base_name = Path(input_path).stem
            generated_dir = Path(input_path).parent / f"{base_name}_ncnn_model"
            
            if generated_dir.exists():
                param_file = generated_dir / "model.ncnn.param"
                bin_file = generated_dir / "model.ncnn.bin"
                
                # Renombrar archivos si se especificó un nombre personalizado
                if output_path != f"{base_name}_ncnn":
                    new_param = Path(f"{output_path}.param")
                    new_bin = Path(f"{output_path}.bin")
                    
                    if param_file.exists():
                        param_file.rename(new_param)
                        print(f"Archivo .param guardado como: {new_param}")
                    
                    if bin_file.exists():
                        bin_file.rename(new_bin)
                        print(f"Archivo .bin guardado como: {new_bin}")
                    
                    # Eliminar directorio temporal
                    import shutil
                    shutil.rmtree(generated_dir)
                else:
                    print(f"Archivos NCNN generados en: {generated_dir}")
                    print(f"  - Parámetros: {param_file}")
                    print(f"  - Modelo: {bin_file}")
            
            print(f"\n✅ Conversión exitosa!")
            print(f"Para usar con el script principal:")
            print(f"  python yolo-detection.py --use-ncnn --ncnn-param {output_path}.param --ncnn-bin {output_path}.bin")
            
            return True
            
    except Exception as e:
        print(f"Error durante la conversión: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convertir YOLOv8 a NCNN para ARM/Raspberry Pi")
    parser.add_argument("--input", "-i", type=str, required=True, 
                       help="Ruta al modelo YOLOv8 (.pt)")
    parser.add_argument("--output", "-o", type=str, default="", 
                       help="Nombre base para archivos de salida (default: nombre del input)")
    parser.add_argument("--imgsz", type=int, default=320, 
                       help="Tamaño de imagen para conversión (default: 320)")
    parser.add_argument("--int8", action="store_true", 
                       help="Usar quantización int8 para mayor velocidad (menor precisión)")
    
    args = parser.parse_args()
    
    # Verificar que el archivo de entrada existe
    if not os.path.exists(args.input):
        print(f"Error: El archivo {args.input} no existe")
        sys.exit(1)
    
    # Generar nombre de salida si no se especificó
    if not args.output:
        input_path = Path(args.input)
        args.output = input_path.stem + "_ncnn"
    
    print("=" * 60)
    print("CONVERTIDOR YOLO a NCNN para ARM/Raspberry Pi")
    print("=" * 60)
    print(f"Modelo de entrada: {args.input}")
    print(f"Nombre base salida: {args.output}")
    print(f"Tamaño de imagen: {args.imgsz}")
    print(f"Quantización int8: {'Sí' if args.int8 else 'No'}")
    print("=" * 60)
    
    success = convert_yolo_to_ncnn(args.input, args.output, args.imgsz)
    
    if success:
        print("\nCONVERSIÓN COMPLETADA ✅")
        print("\nCONSEJOS PARA RASPBERRY PI:")
        print("- Usa resoluciones bajas (256x256 o 320x320) para mejor rendimiento")
        print("- Habilita el monitoreo de temperatura con --monitor-temp")
        print("- Considera usar --frame-skip para videos de alta resolución")
        print("- El formato NCNN debería ser 2-3x más rápido que PyTorch en ARM")
    else:
        print("\nCONVERSIÓN FALLIDA ❌")
        sys.exit(1)


if __name__ == "__main__":
    main()
