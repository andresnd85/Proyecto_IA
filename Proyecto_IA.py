# Proyecto_IA

import cv2

# Cargar imágenes
imagen = cv2.imread('imagen1.jpg')
template = cv2.imread('imagen2.jpg')

# Convertir imágenes a escala de grises
imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Encontrar ubicación en la plantilla
res = cv2.matchTemplate(imagen_gray, template_gray, cv2.TM_SQDIFF)
minimo_val = float('inf')
maximo_val = float('-inf')
minimo_loc = None
maximo_loc = None
x, r, _ = template.shape

# Recorrer la imagen buscando el template
for i in range(imagen.shape[0] - x):
    for j in range(imagen.shape[1] - r):
        # Obtener la región de interés (ROI)
        roi = imagen_gray[i:i+x, j:j+r]
        # Calcular la diferencia entre el template y la ROI
        diff = cv2.absdiff(template_gray, roi)
        # Sumar los valores de diferencia para obtener una medida de similitud
        val = diff.sum()
        # Actualizar la ubicación del mínimo valor de similitud encontrado
        if val < minimo_val:
            minimo_val = val
            minimo_loc = (j, i)

# Calcular las coordenadas de las esquinas del rectángulo
x1, y1 = minimo_loc
x2, y2 = x1 + r, y1 + x

# Dibujar rectángulo en la imagen
cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 300, 0), 3)

# Mostrar imágenes
cv2.imshow('Imagen', imagen)
cv2.imshow('Rectángulo', template)
cv2.waitKey(0)
cv2.destroyAllWindows()
