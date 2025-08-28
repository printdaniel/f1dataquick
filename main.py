import fastf1
import matplotlib.pyplot as plt
import seaborn as sns
from setup_fastf1 import setup_fastf1_cache

# Configurar cache
setup_fastf1_cache()



# Configurar cache
setup_fastf1_cache()


def elegir_gp(year: int):
    """Muestra el calendario de un año y permite elegir GP"""
    schedule = fastf1.get_event_schedule(year)
    print(f"\n--- Calendario {year} ---")
    for idx, row in schedule.iterrows():
        print(f"[ {row['RoundNumber']:2d} ] {row['EventName']} - {row['EventDate'].date()}")

    while True:
        try:
            ronda = int(input("Elige el número de ronda (ej: 1, 2, 3...): "))
            if ronda in schedule["RoundNumber"].values:
                evento = schedule.loc[schedule["RoundNumber"] == ronda].iloc[0]
                return evento
            else:
                print("❌ Ronda inválida, intenta de nuevo.")
        except ValueError:
            print("❌ Ingresa un número válido.")


def elegir_sesion(evento):
    """Permite elegir tipo de sesión disponible"""
    sesiones_validas = {
        "FP1": "FP1",
        "FP2": "FP2",
        "FP3": "FP3",
        "Q": "Qualifying",
        "R": "Race"
    }
    print("\n--- Sesiones disponibles ---")
    for k, v in sesiones_validas.items():
        print(f"{k:4s} → {v}")

    while True:
        sesion = input("Elige sesión (FP1, FP2, FP3, Q, R): ").upper()
        if sesion in sesiones_validas:
            return sesiones_validas[sesion]
        else:
            print("❌ Sesión inválida. Intenta de nuevo.")


def accion_comparar_pilotos():
    """Comparar ritmo entre pilotos en una sesión"""
    year = int(input("Año de la temporada (ej: 2025): "))
    evento = elegir_gp(year)
    sesion_tipo = elegir_sesion(evento)

    print(f"\nCargando datos: {evento['EventName']} {year} - {sesion_tipo}...")
    session = fastf1.get_session(year, int(evento["RoundNumber"]), sesion_tipo)
    session.load()

    # Pedir pilotos
    while True:
        pilotos = input("Introduce códigos de pilotos separados por coma (mínimo 2, ej: VER,LEC,HAM): ")
        pilotos = [p.strip().upper() for p in pilotos.split(",") if p.strip()]
        if len(pilotos) >= 2:
            break
        else:
            print("⚠️ Debes ingresar al menos 2 pilotos.")

    # Filtrar vueltas rápidas de esos pilotos
    laps = session.laps.pick_drivers(pilotos).pick_quicklaps()

    # Gráfico
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=laps, x="Driver", y="LapTime", palette="Set2")
    plt.title(f"Comparación de ritmo - {evento['EventName']} {year} - {sesion_tipo}")
    plt.ylabel("Tiempo de vuelta")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def accion_piloto_individual():
    """Ritmo de un piloto específico en una sesión"""
    year = int(input("Año de la temporada (ej: 2025): "))
    evento = elegir_gp(year)
    sesion_tipo = elegir_sesion(evento)

    print(f"\nCargando datos: {evento['EventName']} {year} - {sesion_tipo}...")
    session = fastf1.get_session(year, int(evento["RoundNumber"]), sesion_tipo)
    session.load()

    piloto = input("Código de piloto (ej: VER, HAM, ALO): ").upper()

    laps = session.laps.pick_driver(piloto).pick_quicklaps()

    plt.figure(figsize=(12, 6))
    plt.plot(laps["LapNumber"], laps["LapTime"].dt.total_seconds(),
             marker="o", label=piloto)
    plt.title(f"Ritmo de {piloto} - {evento['EventName']} {year} - {sesion_tipo}")
    plt.xlabel("Número de vuelta")
    plt.ylabel("Tiempo de vuelta (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def salir():
    print("👋 Saliendo del programa... Hasta la próxima!")


def menu_principal():
    while True:
        print("\n--- Menú Principal ---")
        print("[ 1 ]. Comparar ritmo entre pilotos")
        print("[ 2 ]. Ritmo de un piloto específico")
        print("[ 3 ]. Salir")

        opcion = input("Elige una opción (1, 2, 3): ")

        if opcion == '1':
            accion_comparar_pilotos()
        elif opcion == '2':
            accion_piloto_individual()
        elif opcion == '3':
            salir()
            break
        else:
            print("❌ Opción no válida. Por favor, elige un número del 1 al 3.")
if __name__ == "__main__":
    menu_principal()

