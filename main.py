import fastf1
import matplotlib.pyplot as plt
import seaborn as sns
from setup_fastf1 import setup_fastf1_cache

# Configurar cache
setup_fastf1_cache()

def plot_race_pace(year=2025, gp="Australia"):
    # Cargar sesión de carrera
    session = fastf1.get_session(year, gp, "R")
    session.load()

    # Obtener datos de vueltas
    laps = session.laps.pick_quicklaps()

    # Agrupar tiempos medios por piloto
    avg_laps = laps.groupby("Driver")["LapTime"].mean().sort_values()

    # Gráfico
    plt.figure(figsize=(10,6))
    sns.barplot(x=avg_laps.index, y=avg_laps.values, palette="coolwarm")
    plt.title(f"Ritmo promedio - {gp} {year}")
    plt.ylabel("Tiempo medio de vuelta (s)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_race_pace()

