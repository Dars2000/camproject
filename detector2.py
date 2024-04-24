import cv2
import torch
import time
import json

# Optimizar el uso de OpenCV
cv2.setUseOptimized(True)

model = torch.hub.load("ultralytics/yolov5", "yolov5n", pretrained=True)

def obtener_segundos(num_detecciones):
    if num_detecciones > 30:
        return 50
    elif num_detecciones > 20:
        return 35
    elif num_detecciones > 10:
        return 30
    elif num_detecciones > 5:
        return 20
    elif num_detecciones > 2:
        return 15
    elif num_detecciones > 0:
        return 10
    elif num_detecciones == 0:
        return 0
def obtener_resultado():
    num_resultado = obtener_segundos()
    return num_resultado

def detector():
    cap = cv2.VideoCapture("carros.mp4")
    
    start_time = time.time()
    
    while cap.isOpened():
        status, frame = cap.read()

        if not status:
            break
        
        # Inferencia
        pred = model(frame)
        
        # Filtrar detecciones con confianza mayor o igual a 0.5
        df = pred.pandas().xyxy[0]
        df = df[df["confidence"] >= 0.5]
        
        # Mostrar número de nuevas detecciones
        num_new_detections = df.shape[0]
        print(f"Nuevas detecciones: {num_new_detections}")
        segundos = obtener_segundos(num_new_detections)
        print(f"Segundos: {segundos}")
        
        with open("segundos.json", "w") as json_file:
            json.dump({"segundos": segundos}, json_file)
        #recibidor_segundos.escribir_segundos(segundos)

        # Mostrar conteo de vehículos cada 10 segundos
        elapsed_time = time.time() - start_time
        if elapsed_time >= 10:
            start_time = time.time()  # Actualizar el tiempo de inicio
        
        # Dibujar rectángulos y etiquetas en el frame
        for i in range(df.shape[0]):
            bbox = df.iloc[i][["xmin", "ymin", "xmax", "ymax"]].values.astype(int)
            
            cv2.rectangle(img=frame, pt1=(bbox[0], bbox[1]), pt2=(bbox[2], bbox[3]), color=(255,0,0), thickness=1)
            cv2.putText(frame, 
                        f"{df.iloc[i]['name']}: {round(df.iloc[i]['confidence'], 4)}",
                        (bbox[0], bbox[1] - 15),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        (255,255,255),
                        2)
            
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
    cap.release()
    
    #return segundos 
if __name__ == '__main__':
    detector()
