from tkinter import *
from tkinter import filedialog, messagebox
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
from skimage import io, transform
from sklearn import preprocessing
import joblib
import cv2
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import PassiveAggressiveClassifier
from skimage.transform import resize
from skimage.io import imread
from PIL import Image,ImageTk
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis

global IMG_SIZE, model_folder
IMG_SIZE = (64, 64)
model_folder = 'models'



def uploadDataset():
    global filename, categories
    filename = filedialog.askdirectory(initialdir="Dataset")
    if not filename:
        return

    text.delete('1.0', END)
    text.insert(END, f"Folder Loaded:\n{filename}\n\n")

    categories = [
        d for d in os.listdir(filename)
        if os.path.isdir(os.path.join(filename, d))
    ]

    text.insert(END, "Subfolders found:\n")
    for label in categories:
        text.insert(END, f"- {label}\n")


# This function extracts HOG, LBP, and HSV features from cloud images.
from skimage.feature import hog, local_binary_pattern
IMG_SIZE = (64, 64)

def extract_features(img):
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    pixels = gray.flatten().astype(np.float32)

    # Statistical features
    mean_val = np.mean(pixels)  # Average brightness of the image
    std_val = np.std(pixels)  # Measures contrast
    skew_val = skew(pixels)  # Measures asymmetry of pixel intensity distribution
    kurt_val = kurtosis(pixels) # Measures tailedness of distribution (extreme pixel or not)
    energy_val = np.mean(pixels ** 2) # Measures overall intensity strength

    # GLCM
    glcm = graycomatrix(
        gray,
        distances=[1],
        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
        levels=256,
        symmetric=True,
        normed=True
    )

    contrast = graycoprops(glcm, 'contrast').mean() # Measures local intensity variation
    homogeneity = graycoprops(glcm, 'homogeneity').mean()  # Measures closeness of pixel pairs
    correlation = graycoprops(glcm, 'correlation').mean() # Measures how correlated a pixel is to its neighbor
    dissimilarity = graycoprops(glcm, 'dissimilarity').mean() # Similar to contrast, but linear
    asm = graycoprops(glcm, 'ASM').mean() # Measures uniformity

    return np.array([
        mean_val, std_val, skew_val, kurt_val,
        energy_val, contrast, dissimilarity,
        homogeneity, correlation, asm
    ], dtype=np.float32)



def preprocessing():
    global X,Y
    X_file = os.path.join(model_folder, "X.npy")
    Y_file = os.path.join(model_folder, "Y.npy")

    if os.path.exists(X_file) and os.path.exists(Y_file):
        X = np.load(X_file)
        Y = np.load(Y_file)
        text.insert(END,"X and Y arrays loaded successfully.\n")
    else:
        X = []  # input array
        Y = []  # output array
        for root, dirs, directory in os.walk(filename):
            for j in range(len(directory)):
                name = os.path.basename(root)
                print(f'Loading category: {dirs}')
                print(name+" "+root+"/"+directory[j])
                
                if 'Thumbs.db' not in directory[j]:
                    img_array = cv2.imread(root+"/"+directory[j])
                    
                    # Resize image
                    img_resized = cv2.resize(img_array, IMG_SIZE)

                    # Extract HOG + LBP + HSV features
                    features = extract_features(img_resized)

                    # Append feature vector to X
                    X.append(features)
                    
                    # Append label to Y
                    Y.append(categories.index(name))

        X = np.array(X)
        Y = np.array(Y)
        np.save(X_file, X)
        np.save(Y_file, Y)
    text.insert(END, f"Feature Matrix: {X.shape}\n\n")

def train():
    global x_train, x_test, y_train, y_test

    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split

    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=77, stratify=Y
    )

    text.insert(END, f"X_train shape: {x_train.shape}\n\n")
    text.insert(END, f"y_train shape: {y_train.shape}\n\n")

# ──────────────────────────────────────────────────────────────
# Global containers
# ──────────────────────────────────────────────────────────────

global metrics_overall, class_metrics_storage
metrics_overall = []                    # Overall metrics
class_metrics_storage = {}              # Key = class name → list of dicts (one per model)
def calculateMetrics(algorithm, predict, testY):
    
    testY = testY.astype('int')
    predict = predict.astype('int')
    
    # ── Overall metrics ─────────────────────────────────────
    acc  = accuracy_score(testY, predict) * 100
    prec = precision_score(testY, predict, average='macro', zero_division=0) * 100
    rec  = recall_score(testY, predict, average='macro', zero_division=0) * 100
    f1   = f1_score(testY, predict, average='macro', zero_division=0) * 100
    
    metrics_overall.append({
        'Model': algorithm,
        'Accuracy':  round(acc, 2),
        'Precision': round(prec, 2),
        'Recall':    round(rec, 2),
        'F1-Score':  round(f1, 2)
    })
    
    text.insert(END, f"\n=== {algorithm} Overall ===\n")
    text.insert(END, f"Accuracy  : {acc:.2f}%\n")
    text.insert(END, f"Precision : {prec:.2f}%\n")
    text.insert(END, f"Recall    : {rec:.2f}%\n")
    text.insert(END, f"F1-Score  : {f1:.2f}%\n")
    
    # ── Class-wise metrics ──────────────────────────────────
    report = classification_report(testY, predict, target_names=categories,
                                  output_dict=True, zero_division=0)
    text.insert(END, "\n=== Classification Report ===")
    text.insert(END, classification_report(testY, predict, target_names=categories, zero_division=0))

    for cls in categories:
        # Initialize list for this class if first time
        if cls not in class_metrics_storage:
            class_metrics_storage[cls] = []
        
        class_metrics_storage[cls].append({
            'Model':     algorithm,
            'Precision': round(report[cls]['precision'] * 100, 2),
            'Recall':    round(report[cls]['recall'] * 100, 2),
            'F1-Score':  round(report[cls]['f1-score'] * 100, 2)
        })
    
    # ── Confusion matrix ────────────────────────────────────
    cm = confusion_matrix(testY, predict)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=categories, yticklabels=categories, cbar=False)
    plt.title(f"{algorithm} - Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def model1():
    import os
    import joblib
    from sklearn.ensemble import AdaBoostClassifier
    
    model_path = r"models\adaboost_classifier.pkl"
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        ada_clf = joblib.load(model_path)
        predict = ada_clf.predict(x_test)
        calculateMetrics("AdaBoost Classifier", predict, y_test)
    
    else:
        ada_clf = AdaBoostClassifier(n_estimators=100, random_state=42)
        ada_clf.fit(x_train, y_train)
        predict = ada_clf.predict(x_test)
        joblib.dump(ada_clf, model_path)
        calculateMetrics("AdaBoost Classifier", predict, y_test)
    
def model2():
    from sklearn.naive_bayes import GaussianNB

    model_path = r"models\GNB_classifier.pkl"

    # Ensure folder exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
       gnb_clf = joblib.load(model_path)
       predict = gnb_clf.predict(x_test)
       calculateMetrics("GNB Classifier", predict, y_test)

    else:
        gnb_clf = GaussianNB()
        gnb_clf.fit(x_train, y_train)
        predict = gnb_clf.predict(x_test)
        joblib.dump(gnb_clf, model_path)
        calculateMetrics("GNB Classifier", predict, y_test)


def model3():
    
    from sklearn.neighbors import NearestCentroid
    model_path = r"models\NC_classifier.pkl"

    # Ensure folder exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
       nc_clf = joblib.load(model_path)
       predict = nc_clf.predict(x_test)
       calculateMetrics("NC Classifier", predict, y_test)

    else:
        nc_clf = NearestCentroid()
        nc_clf.fit(x_train, y_train)
        predict = nc_clf.predict(x_test)
        joblib.dump(nc_clf, model_path)
        calculateMetrics("NC Classifier", predict, y_test)

def model4():
    
    from imblearn.ensemble import BalancedRandomForestClassifier
    model_path = r"models\BRF_classifier.pkl"

    # Ensure folder exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
       brf_clf = joblib.load(model_path)
       predict = brf_clf.predict(x_test)
       calculateMetrics("BRF Classifier", predict, y_test)

    else:
        brf_clf = BalancedRandomForestClassifier()
        brf_clf.fit(x_train, y_train)
        predict = brf_clf.predict(x_test)
        joblib.dump(brf_clf, model_path)
        calculateMetrics("BRF Classifier", predict, y_test)


# ----------------------------
# Prediction function
# ----------------------------
def predict_image():
    
    global model_path, pred_idx, pred_label, features, predicted_class, model
    model_path = os.path.join(model_folder, "BRF_classifier.pkl")
    if not os.path.exists(model_path):
        messagebox.showwarning("Warning", "Train model first!")
        return

    model = joblib.load(model_path)

    path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not path:
        return

    # Load image safely
    img_array = cv2.imread(path)
    if img_array is None:
        print(f"Error: Cannot read the image: {path}")
        return None

    # Extract features (resized to IMG_SIZE)
    features = extract_features(img_array).reshape(1, -1)

    # Safety check
    if features.shape[1] != model.n_features_in_:
        raise ValueError(
            f"Feature vector length {features.shape[1]} does not match "
            f"model expected {model.n_features_in_} features."
        )

    # Predict
    pred_idx = model.predict(features)[0]
    pred_label = categories[pred_idx]

    # Display image with predicted label
    plt.figure(figsize=(5,5))
    plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    plt.text(10, 10, f'Predicted: {pred_label}', color='blue', fontsize=12,
             weight='bold', backgroundcolor='white')
    plt.axis('off')
    plt.show()

    return pred_label



# ================= UI =================

main = Tk()
main.title("visual machine learning assessment of meat freshness from image based quality indicators")
main.geometry("1400x900")
bg_image = Image.open("istockphoto-469354734-612x612.jpg")   
bg_image = bg_image.resize((1400, 900))
bg_photo = ImageTk.PhotoImage(bg_image)

bg_label = Label(main, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

font = ('times', 15, 'bold')
font1 = ('times', 13, 'bold')
ff = ('times', 12, 'bold')

Label(main, text="visual machine learning assessment of meat freshness from image based quality indicators",
      bg="gold2", fg="black",
      font=font, height=3, width=120).place(x=0, y=5)

Button(main, text="Dataset", command=uploadDataset, font=ff).place(x=20, y=150)
Button(main, text="Feature Extraction", command=preprocessing, font=ff).place(x=20, y=200)
Button(main, text="Train Test Split", command=train, font=ff).place(x=20, y=250)
Button(main, text="AdaBoostClassifier", command=model1, font=ff).place(x=20, y=300)
Button(main, text="GaussianNB", command=model2, font=ff).place(x=20, y=350)
Button(main, text="NearestCentroid", command=model3, font=ff).place(x=20, y=400)
Button(main, text="BalancedRandomForestClassifier", command=model4, font=ff).place(x=20, y=450)
Button(main, text="Prediction", command=predict_image, font=ff).place(x=20, y=500)


text = Text(main, height=25, width=100, font=font1)
scroll = Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=330, y=100)

main.mainloop()

