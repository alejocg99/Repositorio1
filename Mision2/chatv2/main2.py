from chatbot.data import training_data
from chatbot.model import build_and_train_model, load_model, predict_answers


def chat(model, vectorizer, unique_answers):
    # inicia el modo conversación
    # Mostrar un mensaje inicial al usuario
    print("\n 🤖 Chatbot supervisado listo. Escribe 'salir' para terminar. \n")

    while True:
        # Pedimos una frase al usuario
        user = input(
            " ✍️   Cliente: "
        ).strip()  # strip elimina espacios al inicio y final
        if user.lower() == "":
            print(" 🤖 Agente BOT:" "¡Hasta pronto! \n\n")
            break

        # modelo predice la respuesta
        response = predict_answers(model, vectorizer, unique_answers, user)

        # Mostrar la respuesta en pantalla
        print(" \n 🤖 Agente BOT: ", response, "\n")

def main():
    model, vectorizer, unique_answer = load_model
    # menu principal
    while True:
        print("n=== 🤖 MENU PRINCIPAL DEL CHATBOOT===")
        print("1️⃣ Chatear con el modelo")
        print("2️⃣ Reentrenar el modelo")
        print("3️⃣Salir")
        opcion=input("\n Elige una opcion (1-3);").strip()
        if opcion == "1":
            if model is None:
                print("\n ⚠️ No hay modelo enytrenado. Entrenalo primero.")
            else:
                chat(model,vectorizer,unique_answer)
        elif opcion == "2":
            print("\n 🔁 Reentrenar el modelo con los nuevos datos...")
            model,vectorizer,unique_answers=build_and_train_model(training_data)
            print("✅ Modelo actualizado correctamente.")
        elif opcion == "3":
            print("\n 👋 ¡Hasta luego!")
            break
        else:
            print("\n ✖️ Opcion no valido. intenta nuevamente.")
if __name__=="_main_":
    main()

