import cv2
import face_recognition
import os

# Импортируем все модули

cap = cv2.VideoCapture(0)
# Методом VideoCapture мы получаем видео, тут можно указать путь к mp4 файлу и прочитать его
# Указав 0 мы получаем видео с ВЕБ КАМЕРЫ

image_to_recognition = face_recognition.load_image_file('face-1.1.jpg')
# Теперь начинаем работать с face_recognition, метод load_image_file получает изображение
# В данном случае то фото рами малека которые мы обрезали второй программой

image_enc = face_recognition.face_encodings(image_to_recognition)[0]
# Тут методом face_encodings мы получаем КОДИРОВКУ ЛИЦА.
# Просто у каждого фото с лицом (да и не только) есть КОДИРОВКА.
# Если у нас есть 2 фото с лицами и если их кодировки совпадают, значит на фото один и тот же человвек

recognizer_cc = cv2.CascadeClassifier('faces.xml')
# Про это уже говорил

# Любое видео, это быстро меняющиеся картинки, от того и создается эффект анимации
# Здесь тот же принцип, в бесконечном цикле мы очень быстро меняем изображения и получаем видео
while True:
    success, img = cap.read()
    # Получаем изображение, которое будем быстро показывать, если изображение не получено success будет равен False

    recognize = recognizer_cc.detectMultiScale(img, scaleFactor=2, minNeighbors=3)
    try:
        if len(recognize) != 0:
            # Если на фото есть лицо, делаем то, что ниже
            print("Лицо нашел")
            unknown_face = face_recognition.face_encodings(img)
            # Получем кодировку неизвестного лица (лица которое на видео)

            compare = face_recognition.compare_faces(image_enc, [unknown_face[0]])

            if compare[0] == True:
                # Если мы зашли сюда, значит лица одинаковые
                print('Совпадение!')
            else:
                print('Это не Камыч.')
    except:
        pass
