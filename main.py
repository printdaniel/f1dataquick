import fastf1
import matplotlib.pyplot as plt
import seaborn as sns
from setup_fastf1 import setup_fastf1_cache

# Configurar cache
setup_fastf1_cache()











def menu_principal():
    while True:
        # 1. Mostramos las opciones del menú
        print("\n--- Menú Principal ---")
        print("[ 1 ]. Comparar ritmo de Carrera entre pilotos")
        print("[ 2 ]. Ritmo de carrera de un piloto específico")
        print("[ 3 ]. Salir")

        # 2. Pedimos al usuario que elija una opción
        opcion = input("Elige una opción (1, 2, 3): ")

        # 3. Manejamos la opción elegida con condicionales
        if opcion == '1':
            accion_1()
        elif opcion == '2':
            accion_2()
        elif opcion == '3':
            salir()
            break # Salimos del bucle
        else:
            print("Opción no válida. Por favor, elige un número del 1 al 3.")

if __name__ == "__main__":
    menu_principal()

