import csv
import pandas as pd

def read_ground_truth(file_path):
    """Lee el archivo CSV de ground truth con formato [time,in,out]
    
    Args:
        file_path: Ruta al archivo CSV
        
    Returns:
        Lista de tuplas (tiempo, in, out)
    """
    try:
        # Usar pandas para leer el CSV
        df = pd.read_csv(file_path)
        
        # Verificar que el dataframe tiene las columnas necesarias
        if set(['time', 'in', 'out']).issubset(df.columns):
            # Convertir a lista de tuplas
            ground_truth = list(zip(df['time'].astype(float), 
                                    df['in'].astype(int), 
                                    df['out'].astype(int)))
            return sorted(ground_truth, key=lambda x: x[0])
        else:
            print(f"Error: El CSV debe tener las columnas 'time', 'in', 'out'")
            return []
            
    except Exception as e:
        print(f"Error al leer archivo de ground truth: {e}")
        return []

def evaluate_counting(ground_truth, detected_time, detected_in, detected_out):
    """Evalúa el conteo detectado contra el ground truth
    
    Args:
        ground_truth: Lista de tuplas (tiempo, in, out)
        detected_time: Tiempo actual de detección en segundos
        detected_in: Conteo 'in' detectado
        detected_out: Conteo 'out' detectado
        
    Returns:
        Diccionario con métricas MAE y % de error
    """
    if not ground_truth:
        return None
    
    # Encontrar el punto de ground truth más cercano al tiempo actual
    closest_gt = None
    min_diff = float('inf')
    
    for gt_time, gt_in, gt_out in ground_truth:
        diff = abs(gt_time - detected_time)
        if diff < min_diff:
            min_diff = diff
            closest_gt = (gt_time, gt_in, gt_out)
    
    if closest_gt:
        gt_time, gt_in, gt_out = closest_gt
        
        # Calcular MAE (Error Absoluto Medio)
        mae_in = abs(gt_in - detected_in)
        mae_out = abs(gt_out - detected_out)
        mae_total = abs((gt_in + gt_out) - (detected_in + detected_out))
        
        # Calcular Error Porcentual
        if gt_in > 0:
            error_pct_in = (mae_in / gt_in) * 100
        else:
            error_pct_in = 0 if detected_in == 0 else 100
            
        if gt_out > 0:
            error_pct_out = (mae_out / gt_out) * 100
        else:
            error_pct_out = 0 if detected_out == 0 else 100
            
        if (gt_in + gt_out) > 0:
            error_pct_total = (mae_total / (gt_in + gt_out)) * 100
        else:
            error_pct_total = 0 if (detected_in + detected_out) == 0 else 100
        
        return {
            'gt_time': gt_time,
            'gt_in': gt_in,
            'gt_out': gt_out,
            'mae_in': mae_in,
            'mae_out': mae_out,
            'mae_total': mae_total,
            'error_pct_in': error_pct_in,
            'error_pct_out': error_pct_out,
            'error_pct_total': error_pct_total
        }
    
    return None
