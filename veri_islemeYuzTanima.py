# Bu kod trainer.yml ve labels dosyalarını oluşturur
# Bunlar giriş kodunda kullanılarak kıyaslamayı mümkün kılar.
import os
import numpy as np
from PIL import Image
import cv2
import pickle
import face_recognition

# Yüz tespiti için kullanacağımız Classifier'ın yolunu koda belirtiyoruz.
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# OpenCV kütüphanesinde bulunan LBPH (Local Binary Pattern Histogram) yüz tanıyıcı kullanıyoruz.
#recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Bulunan dosya yolunu tespit edip images klasörüne ulaşılır
baseDir = os.path.dirname(os.path.abspath(__file__))
imageDir = os.path.join(baseDir, "images")
# Kaydını yaptığımız kullanıcıların adlarını tutan liste.
names=["None"]


currentId = 1
labelIds = {}
yLabels = []
xTrain = []

# Bulduğu her bir görüntüyü tek tek gezer ve bu görüntüleri NumPy dizisine dönüştürür
for root, dirs, files in os.walk(imageDir):
    label = os.path.basename(root)
    print(label)
    # Kaydını yaptığımız kullanıcıların adlarını 'names' listesine ekliyoruz.
    names.append(label)
    print(root, dirs, files)
    for file in files:
        print("file",file)
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            #label = os.path.basename(root)
            #print(names)

            if not label in labelIds:
                labelIds[label] = currentId
                #print(labelIds)
                currentId += 1

            # Doğru görüntülere sahip olduğumuzdan emin olmak için yüz algılamayı tekrar gerçekleştiriyoruz.
            # Ve sonra kıyaslama verilerini hazırlıyoruz
            id_ = labelIds[label]
            pilImage = Image.open(path).convert("L")
            imageArray = np.array(pilImage, "uint8")
            faces = faceCascade.detectMultiScale(imageArray)
            # Bulunan yüzlerin x, y koordinatı ve width(genişlik), height(uzunluk) bilgilerini alıyoruz.
            for (x, y, w, h) in faces:
                roi = imageArray[y:y + h, x:x + w]
                xTrain.append(roi)
                yLabels.append(id_)
# Dizin adlarını ve etiket kimliklerini içeren sözlüğü saklıyoruz.
with open("labels", "wb") as f:
    pickle.dump(labelIds, f)
    f.close()
# Verileri işliyoruz ve dosyayı kaydediyoruz.
recognizer.train(xTrain, np.array(yLabels))
recognizer.save("trainer.yml")
#print(labelIds)
names.pop(1)
print(names)

#Yüz Tanıma Aşaması

_id = 0

# Oluşturduğumuz etiket dosyasını açıyoruz ve yüklüyoruz
with open('labels', 'rb') as f:
    dicti = pickle.load(f)
    f.close()

# Kamera kullanacağımızı belirtiyoruz
camera = cv2.VideoCapture(0)

# Kameranın boyut ve çözünürlük ayarlarını yapıyoruz.
camera.set(3, 640)
camera.set(4, 480)
minW = 0.1 * camera.get(3)
minH = 0.1 * camera.get(4)

path = os.path.dirname(os.path.abspath(__file__))
# Yüz tespiti için kullanacağımız Classifier'ın yolunu koda belirtiyoruz.
faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# OpenCV paketinde bulunan LBPH (Local Binary Pattern Histogram) yüz tanıyıcı kullanıyoruz.
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Oluşturulan eğitici dosyayı açıyoruz
recognizer.read("trainer.yml")

# Yazı tipi belirleme
font = cv2.FONT_HERSHEY_SIMPLEX

# Sonsuz Döngü
while True:
    # Kameradan gelen görüntüler okunuyor.
    ret, im = camera.read()

    # Gelen görüntüler renkliden griye çevriliyor.
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Görüntülerde ki yüzler tespit ediliyor.
    faces = faceCascade.detectMultiScale(gray, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        # Yüzlerin etrafına dikdörtgen çiziliyor
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
        _id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        # Güven katsayısının 100'den küçük olması gerekiyor. 0 olması en iyi eşleşmeyi temsil eder
        if confidence < 60:
            if _id!=0:
                print("Tehlike")
                
            # Kullanıcı ismi tespit edilip yerine koyulur
            _id = names[_id]
            # Güven hesaplanır
            confidence = "  {0}%".format(round(100 - confidence))
            print(_id)
        else:
            # Tespit edilemeyen yüzler için 'Unknown'(Bilinmeyen) yazılır
            _id = "unknown"
            # Güven hesaplanır
            confidence = "  {0}%".format(round(100 - confidence))
            print("Unknown")
            
        # Yüzün tespit edildiği yere isim ve güven yüzdesi yazılır
        cv2.putText(im, str(_id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(im, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    # Kamera açık tutulur
    cv2.imshow('camera', im)

    # Programın sonlanması için 'ESC' tuşuna basılması beklenir
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

camera.release()
cv2.destroyAllWindows()
