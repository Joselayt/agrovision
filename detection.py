from ultralytics import YOLO
import cv2
import numpy as np
import os
import torch.nn as nn
import joblib
import torch
import pandas  as pd



model_path = "/home/lyane/Desktop/models/yolov8m.pt"
model = YOLO(model_path)


class Detection(nn.Module):
    def __init__(self, vector_path):
        super().__init__()
        self.fc1 = nn.Linear(7, 5)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(5, 3)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(3, 2)
        self.estado = ['Mal estado', 'Buen estado']
        
        self.vector = joblib.load(vector_path)            

    def forward(self, x:torch.Tensor) -> list:
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = torch.argmax(x, dim=1)
        x = [self.estado[i] for i in x]
        return x

    def process(self, img:np.ndarray) -> list:

        
        # colores en hsv
        n_l = np.array([5, 100, 100])
        n_s = np.array([15, 255, 255])

        v_l = np.array([35, 40, 40])
        v_s = np.array([15, 255, 255])

        r1_l = np.array([0, 70, 50])
        r1_s = np.array([10, 255, 255])

        r2_l = np.array([170, 70, 50])
        r2_s = np.array([180, 255, 255])



        result = model(img, verbose=False)
        h, w = img.shape[:2]

        res = []

        for r in result:
            if len(r.boxes) == 0:
                continue
            for b in r.boxes:
                    
                cls = b.cls.item()
                nombre = model.names[cls].lower().strip()
                if nombre == "orange" or nombre == "apple":
                    xyxy = list(iter([int(c.item()) for c in b.xyxy[0]]))
                    x1,y1,x2,y2 = xyxy
                    recorte = img[y1:y2, x1:x2]
                    total_pixel = recorte.shape[0] * recorte.shape[1]
                    area = ((x2-x1)*(y2-y1)) / (recorte.shape[0] * recorte.shape[1])
                    peri = (((x2-x1) + (y2-y1)) * 2) / ((recorte.shape[0] * recorte.shape[1])*2)

                    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
                    
                    kernel = np.ones((5, 5), dtype=np.uint8)
                    naranja = cv2.inRange(hsv,n_l , n_s)
                    naranja = cv2.morphologyEx(naranja, cv2.MORPH_OPEN, kernel)

                    verde = cv2.inRange(hsv, v_l, v_s)
                    verde = cv2.morphologyEx(verde, cv2.MORPH_OPEN, kernel)

                    rojo_1 = cv2.inRange(hsv, r1_l, r1_s)
                    rojo_1 = cv2.morphologyEx(rojo_1, cv2.MORPH_OPEN, kernel)

                    rojo_2 = cv2.inRange(hsv, r2_l, r2_s)
                    rojo_2 = cv2.morphologyEx(rojo_2, cv2.MORPH_OPEN, kernel)

                    porcent_naranja = cv2.countNonZero(naranja) / total_pixel
                    porcent_verde = cv2.countNonZero(verde) / total_pixel
                    porcent_rojo_1 = cv2.countNonZero(rojo_1)
                    porcent_rojo_2 = cv2.countNonZero(rojo_2)
                    porcent_rojo = (porcent_rojo_1 + porcent_rojo_2) / total_pixel

                    res.append({
                        'nombre':nombre,
                        'area':area,
                        'perimetro':peri,
                        'conf':b.conf.item(),
                        'colores':{
                            'verde':porcent_verde,
                            'naranja':porcent_naranja,
                            'rojo':porcent_rojo
                        },
                        'xyxy':xyxy
                    })  


        if len(res) > 0:
            return res
        else:
            return False


    def response(self, frame, byte:bool =False) -> np.ndarray:

        if isinstance(frame, bytes):    
            frame = np.frombuffer(frame, dtype=np.uint8)
            frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        elif isinstance(frame, np.ndarray):
            pass
        else:
            raise TypeError(f'Expected IMG: numpy.ndarray or bytes not {type(img)}')


        P = self.process(frame)
        datos = []
        if P:
                
            res = {}
            for r in P:
                datos.append((r['nombre'], r['conf']))
                res.setdefault('nombre', []).append(r['nombre'])
                res.setdefault('area', []).append(r['area'])
                res.setdefault('perimetro', []).append(r['perimetro'])
                res.setdefault('verde', []).append(r['colores']['verde'])
                res.setdefault('rojo', []).append(r['colores']['rojo'])
                res.setdefault('naranja', []).append(r['colores']['naranja'])

            df = pd.DataFrame(res)
            nombre = df['nombre']
            nombre = self.vector.transform(nombre.values).toarray()
            nombre = pd.DataFrame(nombre, columns=[f'n{i}' for i in range(nombre.shape[1])])
            df = df.drop(columns=['nombre'])
            df['n0'] = nombre['n0']
            df['n1'] = nombre['n1']

            salida = self.forward(torch.tensor(df.values, dtype=torch.float32))

            respuesta = []
            for resultado, diagnostico in zip(salida, P):

                respuesta.append(
                    {
                        'nombre':diagnostico['nombre'],
                        'estado':resultado,
                        'xyxy':diagnostico['xyxy']
                    }
                )
            
            for fruta in respuesta:
                x0,y0,x1,y1 = fruta['xyxy']

                cv2.rectangle(frame, (x0,y0), (x1, y1), (0,0,128), 2)
                texto = "Fruta: {fruta}; Estado: {estado}".format(fruta= 'Manzana' if fruta['nombre']=="apple" else 'Naranja', estado=fruta['estado'])
                cv2.putText(frame, texto.split(";")[0].strip(), (x0+10, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,0,0),2)
                cv2.putText(frame, texto.split(";")[1].strip(), (x0+10, y0 + 60), cv2.FONT_HERSHEY_DUPLEX, 0.5, (128,50,50),2)

        if not byte:        
            return {'img':frame, 'datos':datos}
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            buffer = buffer.tobytes()
            return {'img':buffer, 'datos':datos}
    

