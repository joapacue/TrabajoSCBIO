
##### Bibliotecas a importar ######

#Imports necesarios para la red
import numpy as np
from pynput.keyboard import Key, Controller
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import Xception
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.applications.xception import preprocess_input, decode_predictions

#Imports necesarios para la captura de imágenes
import cv2
import os
import time
import mediapipe as mp
from keras.models import load_model

#### Carga inicial de variables
x1 = 0
x2 = 0
y1 = 0
y2 = 0
respuesta = 0
dedos_reg = 0
flag = 0


tam = 224

dire_img = ['Mano_abierta', 'Mano_cerrada', 'Mano_pausa'] #Labels de los resultados
clase_manos = mp.solutions.hands
manos = clase_manos.Hands()
dibujo = mp.solutions.drawing_utils

cnn = load_model('Modelos/MobileNetV2_modificado_DataSet_224_Triple_V1.h5')  #Cargamos el modelo


#### Abriendo e inicializando cámara
cap = cv2.VideoCapture(0)
#### Inicializando salida por teclado
kb=Controller()

if not cap.isOpened():
    print("[ERROR] Error abriendo la cámara, es posible que otra aplicación la esté usando")
    exit()
else:
    # Inicializamos un contador para regular los FPS de la captura
    timestamp = int(time.time())

while True:
    # Leemos un frame de la Webcam
    ret, frame = cap.read()
    color=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#Pasamos de BGR a RGB
    copia=frame.copy()
    resultado = manos.process(color)
    posiciones = []

    if not ret:
        print("[ERROR] Error leyendo frame, aboratando ejecución del programa")
        break

    if resultado.multi_hand_landmarks: #Si hay algo en los resultados entramos al if

        for mano in resultado.multi_hand_landmarks:  #Buscamos la mano dentro de la lista de manos que nos da el descriptor
            for id, lm in enumerate(mano.landmark):  #Vamos a obtener la informacion de cada mano encontrada por el ID
                alto, ancho, c = frame.shape
                corx, cory = int(lm.x*ancho), int(lm.y*alto) #Extraemos la ubicacion de cada punto que pertence a la mano en coordenadas
                posiciones.append([id, corx, cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)

        if len(posiciones) != 0:
                pto_i5 = posiciones[9]  # Punto central
                x1, y1 = (pto_i5[1] - round(tam / 2)), (pto_i5[2] - round(tam / 2))  # Obtenemos el punto incial y las longitudes
                if (x1 <= 0):
                    x1 = 0
                if (y1 <= 0):
                    y1 = 0
                x2, y2 = x1 + tam, y1 + tam

                dedos_reg = copia[y1:y2, x1:x2]
                dedos_reg = cv2.resize(dedos_reg, (tam, tam), interpolation=cv2.INTER_CUBIC)  # Redimensionamos las fotos
                flag = 1
    else:
         flag = 0
         dedos_reg = 0
    

    ### Leemos el tiempo actual en millis y sólo llamamos a la red 2 veces por seg
    current_timestamp = time.time()
    if (current_timestamp > timestamp+1) & flag:
                x = tf.keras.preprocessing.image.img_to_array(dedos_reg)  # Convertimos la imagen a una matriz
                x = np.expand_dims(x, axis=0)  # Agregamos nuevo eje
                x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
                vector = cnn.predict(x)  # Va a ser un arreglo de 2 dimensiones, donde va a poner 1 en la clase que crea correcta
                resultado = vector[0]  # ej:[0 0 1]
                respuesta = np.argmax(resultado)  # Nos entrega el indice del valor mas alto 0-6

                print('[DEBUG]', resultado, dire_img[respuesta]) # Mostramos el resultado

                ### Dependiendo del resultado que saque la red presionamos la flecha de la derecha o de la izq o ninguna

                if(respuesta == 0):#Mano_abierta
                     kb.press(Key.right)
                     kb.release(Key.right)
                elif(respuesta == 1):#Mano_cerrada
                     kb.press(Key.left)
                     kb.release(Key.left)
                # Mano_pausa

                timestamp = time.time()

    if flag:
        colores = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), colores[respuesta], 3)
        cv2.putText(frame, '{}'.format(dire_img[respuesta]), (x1, y1 - 5), 1, 1.3, colores[respuesta], 1, cv2.LINE_AA)

    cv2.imshow("Webcam", frame)


    #### Abortamos la ejecución del programa al apretar la tecla Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Tecla Q presionada, abortando ejecución del programa")
        cap.release()
        cv2.destroyAllWindows()
        break
    
