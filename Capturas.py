import cv2
import mediapipe as mp
import os

#----------------------------- Creamos la carpeta donde almacenaremos el entrenamiento ---------------------------------
nombre = 'Mano_pausa'#Mano_cerrada Mano_abierta Mano_pausa
direccion = 'C:/Users/alexm/OneDrive/Documentos/SCBIO/TrabajoSCBIO/DataSet_224_Expandido'
carpeta = direccion + '/' + nombre
if not os.path.exists(carpeta):
    print('Carpeta creada: ', carpeta)
    os.makedirs(carpeta)

#Asignamos un contador para el nombre de la fotos
num_im = 25
maximo = num_im*10
tam = 224
cont = 0
#Leemos la camara
cap = cv2.VideoCapture(0)

#----------------------------Creamos un obejto que va almacenar la deteccion y el seguimiento de las manos------------
clase_manos = mp.solutions.hands
manos = clase_manos.Hands() #Primer parametro, FALSE para que no haga la deteccion 24/7
                            #Solo hara deteccion cuando hay una confianza alta
                            #Segundo parametro: numero maximo de manos
                            #Tercer parametro: confianza minima de deteccion
                            #Cuarto parametro: confianza minima de seguimiento

#----------------------------------Metodo para dibujar las manos---------------------------
dibujo = mp.solutions.drawing_utils #Con este metodo dibujamos 21 puntos criticos de la mano


while (1):
    ret, frame = cap.read() #Leemos el frame de la camara
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)#Pasamos de BGR a RGB
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []  # En esta lista vamos a almacenar las coordenadas de los puntos
    #print(resultado.multi_hand_landmarks) #Si queremos ver si existe la deteccion

    if resultado.multi_hand_landmarks: #Si hay algo en los resultados entramos al if
        for mano in resultado.multi_hand_landmarks:  #Buscamos la mano dentro de la lista de manos que nos da el descriptor
            for id, lm in enumerate(mano.landmark):  #Vamos a obtener la informacion de cada mano encontrada por el ID
                #print(id,lm) #Como nos entregan decimales (Proporcion de la imagen) debemos pasarlo a pixeles
                alto, ancho, c = frame.shape  #Extraemos el ancho y el alto de los fotpgramas para multiplicarlos por la proporcion
                corx, cory = int(lm.x*ancho), int(lm.y*alto) #Extraemos la ubicacion de cada punto que pertence a la mano en coordenadas
                posiciones.append([id, corx, cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)
            if len(posiciones) != 0:
                pto_i5 = posiciones[9] #Punto central
                x1, y1 = (pto_i5[1]-round(tam/2)), (pto_i5[2]-round(tam/2)) #Obtenemos el punto incial y las longitudes
                x2, y2 = x1 + tam, y1 + tam
                dedos_reg = copia[y1:y2, x1:x2]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            if(cont % 10 == 0):
                dedos_reg = cv2.resize(dedos_reg, (tam, tam), interpolation=cv2.INTER_CUBIC) #Redimensionamos las fotos
                cv2.imwrite(carpeta + "/Mano_{}.jpg".format(round(cont/10)), dedos_reg)#Mano_Alejandro Mano_Andrei Mano_Jaime Mano_Joaquin Mano_Lorenzo Mano_Jesusec
            cont = cont + 1





    cv2.imshow("Video",frame)
    k = cv2.waitKey(1)
    if k == 27 or cont >= maximo:
        break
cap.release()
cv2.destroyAllWindows()