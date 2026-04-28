import os # Trabaja con el sistema operativo
import pickle# Guarda y carga objetos de python en archivos
from sklearn.feature_extraction.text import CountVectorizer
# CountVectorizer convierte texto en un vector
from sklearn.naive_bayes import MultinomialNB
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR,"model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR,"vectorizer.pkl")
ANSWERS_PATH = os.path.join(MODEL_DIR,"answers.pkl")

def buid_and_train_model(train_pairs):
    questions = [q for q, _ in train_pairs]  
    answers = [a for _, a in train_pairs]
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(questions)
    unique_answers = sorted(set(answers))
    answer_to_label ={a: i for i, a in enumerate(unique_answers)}
    y = [answer_to_label[a] for a in answers]
    model = MultinomialNB()
    model.fit(x,y)
    #Crear carpeta para guardar el modelo si no existe
    os.makedirs(MODEL_DIR,exist_ok=True)
    #Guardar los objetos entrenados
    with open(MODEL_PATH,"wb") as f:
        pickle.dump(model,f)
    with open(VECTORIZER_PATH,"wb") as f:
        pickle.dump(vectorizer,f)
    with open(ANSWERS_PATH,"wb") as f:
        pickle.dump(unique_answers,f)
    print("🆗 Modelo entrenado y guardado correctamente")
    return model, vectorizer, unique_answers

def load_model():
    if(
        os.path.exists(MODEL_PATH)
        and os.path.exists(VECTORIZER_PATH)
        and os.path.exists(ANSWERS_PATH)
    ):
        with open(MODEL_PATH,"rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH,"rb") as f:
            vectorizer = pickle.load(f)
        with open(ANSWERS_PATH,"rb") as f:
            unique_answers = pickle.load(f)
        print("📁 Modelo cargado desde disco.")
        return model, vectorizer, unique_answers
    else:
        print("⚠️ No hay modelo guardado. Será necesario entrenarlo")
        return None,None,None

def predict_answer(model,vectorizer,unique_answers,user_text):
    x = vectorizer.transform([user_text])
    label = model.predict(x)[0]
    return unique_answers[label]
