import threading
from collections import Counter
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
from PIL import Image
from deepface import DeepFace
from torch.autograd import Variable as V
from torch.nn import functional as F
from torchvision import transforms as trn
from datetime import datetime


arch = 'resnet18'
model_file = 'resnet18_places365.pth.tar'
model = models.__dict__[arch](num_classes=365)
checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)
model.eval()

# Image transform for Places365
centre_crop = trn.Compose([
    trn.Resize((256, 256)),
    trn.CenterCrop(224),
    trn.ToTensor(),
    trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load scene class labels
file_name = 'categories_places365.txt'
classes = [line.strip().split(' ')[0][3:] for line in open(file_name)]
classes = tuple(classes)

# Mapping of mood to emoji



emotion_counter = Counter()
gender_counter = Counter()
age_list = []
race_counter = Counter()



frame_count = 0
last_mood = None
last_age = None
last_gender = None
last_scene = None
last_race = None

# Thread lock
lock = threading.Lock()

# Enable live plots
plt.ion()


# ------------------- DeepFace Analysis Thread -------------------
def analyze_face_async(frame, actions):
    global last_mood, last_age, last_gender,last_race, result
    try:
        result = DeepFace.analyze(frame, actions=actions, enforce_detection=False)

        # DeepFace can return a list of dicts or a single dict
        if isinstance(result, list):
            result = result[0]

        with lock:
            if 'emotion' in actions:
                mood = result.get('dominant_emotion')
                if mood and mood != '...':
                    last_mood = mood
                    emotion_counter[mood] += 1

            if 'age' in actions:
                age = result.get('age')
                if isinstance(age, (int, float)):
                    last_age = age
                    age_list.append(age)

            if 'gender' in actions:
                gender_result = result.get('gender')

                if isinstance(gender_result, dict):
                    # Use scores to determine majority
                    man_score = gender_result.get('Man', 0)
                    woman_score = gender_result.get('Woman', 0)

                    if man_score > woman_score:
                        last_gender = 'Man'
                        gender_counter['Man'] += 1
                    else:
                        last_gender = 'Woman'
                        gender_counter['Woman'] += 1

                elif isinstance(gender_result, str):
                    # Fallback for string-based gender
                    if gender_result.lower() == 'male':
                        last_gender = 'Man'
                        gender_counter['Man'] += 1
                    elif gender_result.lower() == 'female':
                        last_gender = 'Woman'
                        gender_counter['Woman'] += 1

            if 'race' in actions:
                last_race = result.get('dominant_race')

    except Exception as e:
        print(f"[ERROR] DeepFace async analysis failed: {e}")


# Optional utility function for drawing text
def draw_text_with_outline(img, text, position, font=cv2.FONT_HERSHEY_DUPLEX, font_scale=0.7, text_color=(255, 255, 255), outline_color=(0, 0, 0), thickness=1):
    """Draw text with an outline for better readability."""
    x, y = position
    # Draw outline (thicker black)
    cv2.putText(img, text, (x, y), font, font_scale, outline_color, thickness + 1, cv2.LINE_AA)
    # Draw main text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)


# ------------------- Start Webcam -------------------
cap = cv2.VideoCapture(0)
# Set frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1040)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)


print("[INFO] Webcam started. Press 'q' to quit.")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read from webcam.")
        break
        # Detect faces and count them
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    num_faces = len(faces)  # <-- Define here

    # Now you can use num_faces, e.g., draw or print it
    print(f"[INFO] Number of faces detected: {num_faces}")


    # When drawing text on frame, use num_faces variable here:
    draw_text_with_outline(frame, f"Faces: {num_faces}", (10, 180), font_scale=0.8, text_color=(0, 0, 255), thickness=1)

    # Get current date and time
    now = datetime.now()
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Draw date and time
    draw_text_with_outline(frame, date_time_str, (10, 210), font_scale=0.8, text_color=(50, 205, 50), thickness=1)

    frame_count += 1


    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_img = V(centre_crop(img_pil).unsqueeze(0))
    logit = model.forward(input_img)

    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    scene = classes[idx[0]]
    last_scene = scene


    # if frame_count % 10 == 0:
    #     threading.Thread(target=analyze_face_async, args=(frame.copy(), ['emotion']), daemon=True).start()
    # if frame_count % 30 == 0:
    #     threading.Thread(target=analyze_face_async, args=(frame.copy(), ['race','age', 'gender']), daemon=True).start()

    if frame_count % 40 == 0:
        threading.Thread(target=analyze_face_async, args=(frame.copy(), ['emotion']), daemon=True).start()
    if frame_count % 100 == 0:
    # if frame_count % 30 == 0:   #rupali
        threading.Thread(target=analyze_face_async, args=(frame.copy(), ['age', 'gender']), daemon=True).start()
    if frame_count % 130 == 0:
    # if frame_count % 50 == 0:     #rupali
        threading.Thread(target=analyze_face_async, args=(frame.copy(), ['race']), daemon=True).start()

    with lock:
        emoji = ''
        mood_text = f"Mood: {last_mood or 'Detecting...'}"
        if isinstance(last_age, (int, float)):
            age_range_start = (int(last_age) // 10) * 10
            age_range_end = age_range_start + 10
            age_text = f"Age: {age_range_start}-{age_range_end}"
        else:
            age_text = "Age: Detecting..."

        gender_text = f"Gender: {last_gender or 'Detecting...'}"
        scene_text = f"Scene: {last_scene or 'Detecting...'}"
        race_text = f"Race: {last_race or 'Detecting...'}"



    # Display all texts
    draw_text_with_outline(frame, scene_text, (10, 30), font_scale=0.8, text_color=(0, 0, 139))  # Dark Blue
    draw_text_with_outline(frame, mood_text, (10, 60), font_scale=0.8, text_color=(0, 100, 0))  # Dark Green
    draw_text_with_outline(frame, age_text, (10, 90), font_scale=0.8, text_color=(139, 69, 19))  # Saddle Brown
    draw_text_with_outline(frame, gender_text, (10, 120), font_scale=0.8, text_color=(105, 105, 105))  # Dim Gray
    draw_text_with_outline(frame, race_text, (10, 150), font_scale=0.8, text_color=(72, 61, 139))  # Dark Slate Blue
    draw_text_with_outline(frame, f"Faces: {num_faces}", (10, 180), font_scale=0.8, text_color=(0, 0, 255), thickness=1)

    cv2.imshow("Mood, Age, Gender, Environment Detection", frame)


    if frame_count % 30 == 0 and emotion_counter:
        plt.clf()
        plt.bar(emotion_counter.keys(), emotion_counter.values(), color='skyblue')
        plt.title("Real-time Emotion Frequency")
        plt.xlabel("Emotion")
        plt.ylabel("Count")
        plt.pause(0.001)



    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

















