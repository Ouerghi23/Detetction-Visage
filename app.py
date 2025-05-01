import cv2
import os
from deepface import DeepFace
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore

# Initialiser Firebase
cred = credentials.Certificate("serviceAccount.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Charger les visages de référence
known_faces = {}
for filename in os.listdir("visages"):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join("visages", filename)
        try:
            extracted = DeepFace.extract_faces(img_path=path, enforce_detection=False)
            if extracted:
                name = os.path.splitext(filename)[0]
                known_faces[name] = extracted[0]['face']  # Image numpy
        except Exception as e:
            print(f"Erreur chargement visage {filename}: {e}")

# Démarrer la webcam
cap = cv2.VideoCapture(0)
print("📸 Caméra activée. Appuyez sur 'q' pour quitter.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Erreur d'accès à la caméra.")
        break

    try:
        faces = DeepFace.extract_faces(img_path=frame, enforce_detection=False)
        for face_info in faces:
            face_img = face_info["face"]
            area = face_info["facial_area"]
            x, y, w, h = area["x"], area["y"], area["w"], area["h"]

            name_found = "Inconnu"

            for name, known_face in known_faces.items():
                try:
                    result = DeepFace.verify(img1_path=face_img, img2_path=known_face, enforce_detection=False)
                    if result["verified"]:
                        name_found = name
                        # Enregistrement dans Firebase
                        db.collection("etudiants").document(name).set({"etat": "Présent"}, merge=True)
                        break
                except Exception as e:
                    print(f"Erreur DeepFace.verify avec {name} : {e}")

            # Afficher le cadre et nom
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name_found, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    except Exception as e:
        print(f"Erreur extraction visage : {e}")

    cv2.imshow("🎥 Reconnaissance Faciale Temps Réel", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
