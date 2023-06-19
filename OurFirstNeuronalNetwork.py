# BASED ON THE GREATE TENSORFLOW STARTING TUTORIAL: https://www.tensorflow.org/tutorials/quickstart/beginner
# Author of Alterations: Jonas Sorgalla
# Commentary language in the following is german
import tensorflow as tf
import cv2
import numpy as np

# CHECK ob TensorFlow verfügbar ist
print("TensorFlow version:", tf.__version__)


# HILSFUNKTIONEN
def convertImage(filepath):
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_LINEAR)
    # Wenn die Bilder nicht schwarzen hintergrund und weiße schrift haben, müssen sie noch invertiert werden, z.B. so
    # img_inverted = cv2.bitwise_not(img_resized)
    img_normalized = img_resized / 255.0  # normalize
    img_batch_dimension = tf.expand_dims(img_normalized, axis=0)
    predictions = model(img_batch_dimension).numpy()
    print("Unser Modell denkt zu folgenden Wahrscheinlichkeiten, dass es zu den Klassen 0 - 9 gehört:")
    print(predictions)
    category = np.argmax(predictions, axis=1)
    print(category)
    print("")


# ++++ START DES EIGENTLICH MACHINE LEARNINGS +++++

# ROHDATEN
# Hier laden wir das DataSet MNIST bestehendaus 60000 Trainingsbildern je 28x28 Pixel und 10000 Testbildern
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# DATENAUFBEREITUNG
# Da der Wert jedes Pixels aktuell zwischen 0 (schwarz) und 255 (weiß) liegt, wir aber nur mit Werten
# zwischen 0 und 1 rechnen können, müssen wir diese noch in den passenden Zahlenraum konvertieren
x_train, x_test = x_train / 255.0, x_test / 255.0

# MODELL KONSTRUKTION
# Hier beschreiben wir, aus welchen Schichten unser neuronales Netz bestehen soll!
model = tf.keras.models.Sequential([
    # Schicht 1: Wir starten mit einer "Hilfs"-Schicht, die aus dem 28x28 Pixel Bildern,
    # d.h. einer Matrix, einen Vektor, also eine "einfache" Zahlenreihe macht.
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # Schicht 2: Den nun 128 Zahlen langen Vektor (28x28 -> 784x1) schicken wir in unseren ersten richtigen
    # neuronal network layer. Die Aktivierungsfunktion, also wann feuert eins der 128 Neuronen, ist "Relu"
    # d.h. als max(x, 0) der Inputs festgelegt. D.h. wenn der Input 0.5 ist, dann feuert das Neuron mit 0.5,
    # wenn der Input -0.2 ist, dann feuert das Neuron mit 0.
    tf.keras.layers.Dense(64, activation='relu'),
    # Schicht 3: Dropout ist eine Hilfsschicht, die in diesem Fall 20 % der Werte zufällig auf 0 setzt.
    # Dies ist eine etablierte Strategie, um "Overfitting" zu verhindern
    tf.keras.layers.Dropout(0.2),
    # Schicht 4: Die Ausgaben der 128 Neuronen werden am Ende auf 10 Neuronen zusammengeschrumpft,
    # da wir ja "nur" 10 mögliche Labelausprägungen, man spricht von Klassen, haben (0, 1, 2, 3, ...., 9)
    tf.keras.layers.Dense(10)
])

# LOSS FUNKTION
# Cross Entropy ist die häufigst genutzten Loss-Funktionen
# bei Klassifizierungsproblemen. Der Wert des cross-entropy loss ist gering,
# wenn die errechnete Wahrscheinlichkeit dem echten Label entspricht
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# MODELL FERTIG KONFIGURIEREN
# Hier bauen wir die vorher definierten Angaben für das Modell zusammen
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# MODELL TRAINIEREN
# Jetzt trainieren wir tatsächlich das Modell, indem wir die Trainingsdaten und passende Labels übergeben
# aus Effizienzgründen legen wir jedoch nicht sofort mit allen 60000 Datensätzen los, sondern teilen die
# Testdaten in 32 "batches".
# epochs gibt anschließend an, wie viele der Batches wir nehmen und damit unser Netz trainieren
# epochs = 1 und batch_size=32 führt also dazu, dass wir "nur" 1876 Datensätze für das Training nutzen
# je höher die epochs & batch_size, desto rechenintensiver wird das Ganze natürlich auch
# Da wir bei jedem Epoch die Werte der vorherigen Durchgänge immer feiner "tunen", könnte es sogar sein, dass wir
# bei kleiner Batchsize und vielen Epochs zu falschen Minima bei der Loss-Funktion kommen.
# Am Ende ist es tatsächlich ein Ausprobieren ;-)
model.fit(x_train, y_train, batch_size=32, epochs=5)

# MODELL EVALUATION
# Wir prüfen das trainierte Modell mit den vorher zurückgelegten Testdaten und erhalten eine Angabe
# der Genauigkeit (Accuracy)
model.evaluate(x_test, y_test, verbose=2)
model.summary()

# MODELL AUSLIEFERUNG
# Naja, wir liefern unser Modell jetzt nicht wirklich in Software aus, auch wenn das möglich wäre.
# Stattdessen nutzen wir an dieser Stelle das Modell und füttern es mit "echten" Daten von uns und gucken,
# ob auch hier die Erkennung klappt.
print("-------- NUTZUNG DES MODELLS --------")

print("Vorhersage für null:")
convertImage(r'images/null.jpg')

print("Vorhersage für eins:")
convertImage(r'images/eins.jpg')

print("Vorhersage für zwei:")
convertImage(r'images/zwei.jpg')

print("Vorhersage für drei:")
convertImage(r'images/drei.jpg')

print("Vorhersage für vier:")
convertImage(r'images/vier.jpg')

print("Vorhersage für fünf:")
convertImage(r'images/fuenf.jpg')

print("Vorhersage für sechs:")
convertImage(r'images/sechs.jpg')

print("Vorhersage für sieben:")
convertImage(r'images/sieben.jpg')

print("Vorhersage für acht:")
convertImage(r'images/acht.jpg')

print("Vorhersage für neun:")
convertImage(r'images/neun.jpg')

print("Vorhersage für X:")
convertImage(r'images/x.jpg')

print("Vorhersage für W:")
convertImage(r'images/w.jpg')

