import fastf1
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
import matplotlib.ticker as ticker

# Definir ruta
CACHE_DIR = os.path.join(os.getcwd(), "cache")

# Crear si no existe
os.makedirs(CACHE_DIR, exist_ok=True)

# Habilitar caché
fastf1.Cache.enable_cache(CACHE_DIR)

driver_colors = {
    # Red Bull
    "VER": "#1E5BC6",  # Verstappen - Azul oscuro de Red Bull
    "TSU": "#1E5BC6",  # Tsunoda

    # Ferrari
    "LEC": "#ED1C24",  # Leclerc - Rojo Ferrari
    "HAM": "#ED1C24",  # Hamilton

    # Mercedes
    "RUS": "#00A3B9",  # Russell - Cian/verde azulado de Mercedes
    "ANT": "#00A3B9",  # Antonelli

    # McLaren
    "NOR": "#FF8000",  # Norris - Naranja papaya de McLaren
    "PIA": "#FF8000",  # Piastri

    # Aston Martin
    "ALO": "#2D8266",  # Alonso - Verde británico de Aston Martin
    "STR": "#2D8266",  # Stroll

    # Alpine
    "GAS": "#0093D0",  # Gasly - Azul claro de Alpine
    "COL": "#0093D0",  # Colapinto

    # Williams
    "ALB": "#005AFF",  # Albon - Azul intenso de Williams
    "SAI": "#005AFF",  # Sainz

    # Kick Sauber
    "HUL": "#00FF00",  # Hülkenberg - Verde neón de Kick Sauber
    "BOR": "#00FF00",  # Bortoleto

    # Haas
    "OCO": "#B0B6B8",  # Ocon - Gris plateado de Haas
    "BEA": "#B0B6B8",  # Bearman

    # RB (Racing Bulls)
    "LAW": "#6692FF",  # Lawson - Azul claro de RB
    "HAD": "#6692FF",  # Hadjar
}

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

def cargar_sesion():
    """Solicita año, evento y tipo de sesión, y carga una sesión de FastF1."""
    year = int(input("Año de la temporada (ej: 2025): "))
    evento = elegir_gp(year)
    sesion_tipo = elegir_sesion(evento)
    print(f"\nCargando datos: {evento['EventName']} {year} - {sesion_tipo}...")
    session = fastf1.get_session(year, int(evento["RoundNumber"]), sesion_tipo)
    session.load()
    return session, evento, year, sesion_tipo
# ----------------------------------------------------------------------------
# Comaración entre pilotos
# ----------------------------------------------------------------------------
def accion_comparar_pilotos():
    """Comparar ritmo entre pilotos en una sesión con violin plot (robusto)."""
    session, evento, year, sesion_tipo = cargar_sesion()

    # Pedir pilotos
    while True:
        pilotos = input("Introduce códigos de pilotos separados por coma (mínimo 2, ej: VER,LEC,HAM): ")
        pilotos = [p.strip().upper() for p in pilotos.split(",") if p.strip()]
        if len(pilotos) >= 2:
            break
        else:
            print("⚠️ Debes ingresar al menos 2 pilotos.")

    # Filtrar vueltas rápidas de esos pilotos (FastF1 Laps obj)
    laps = session.laps.pick_drivers(pilotos).pick_quicklaps()

    # Convertir a DataFrame por seguridad
    laps_df = pd.DataFrame(laps)  # si ya es DataFrame, esto lo deja igual

    # --- Diagnóstico rápido (imprime para ver qué hay)
    print("\n--- Diagnóstico de columnas y tipos ---")
    print(laps_df.dtypes)
    print("Pilotos encontrados:", sorted(laps_df['Driver'].unique().tolist()))
    print("Primeras filas:")
    print(laps_df[['Driver','LapTime','LapNumber']].head())

    # Asegurar LapTimeSeconds (float)
    if 'LapTime' not in laps_df.columns:
        raise RuntimeError("No se encontró la columna 'LapTime' en los datos.")

    # Si es timedelta, convertir; si es string, intentar parsear; si es numérico, usarlo.
    if pd.api.types.is_timedelta64_dtype(laps_df['LapTime']):
        laps_df['LapTimeSeconds'] = laps_df['LapTime'].dt.total_seconds()
    else:
        # Intento parsear si viene como string
        try:
            # Convierte strings tipo '0 days 00:01:23.456000' o '00:01:23.456'
            laps_df['LapTime'] = pd.to_timedelta(laps_df['LapTime'])
            laps_df['LapTimeSeconds'] = laps_df['LapTime'].dt.total_seconds()
        except Exception:
            # Por último, intentar forzar numérico (si ya está en segundos)
            laps_df['LapTimeSeconds'] = pd.to_numeric(laps_df['LapTime'], errors='coerce')

    # Quitar filas sin tiempo
    laps_df = laps_df.dropna(subset=['LapTimeSeconds'])
    if laps_df.empty:
        raise RuntimeError("No quedan vueltas con tiempo válido después del filtrado.")

    # Asegurar que la columna Driver sea string y los códigos estén en mayúsculas
    laps_df['Driver'] = laps_df['Driver'].astype(str).str.upper()

    # Forzar orden de pilotos (en el mismo orden que los ingresados por el usuario, si están presentes)
    present_drivers = [d for d in pilotos if d in laps_df['Driver'].unique()]
    if not present_drivers:
        present_drivers = sorted(laps_df['Driver'].unique())
    order = present_drivers

    # Preparar paleta: si tienes driver_colors, generar lista en orden
    # Si no existe driver_colors, usar palette 'Set2'
    try:
        palette_list = [driver_colors.get(d, "#888888") for d in order]
    except Exception:
        palette_list = None

    # Crear carpeta para figuras
    out_dir = "output/figures"
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{out_dir}/violin_comparacion_{evento['EventName'].replace(' ','_')}_{year}_{sesion_tipo}.png"

    # SOLUCIÓN: Usar solo un estilo y configurar explícitamente
    plt.style.use("default")  # Resetear a estilo por defecto
    sns.set_theme(style="darkgrid")  # Seaborn maneja el estilo

    fig, ax = plt.subplots(figsize=(12, 7))

    # Verificar que tenemos datos para cada piloto
    print(f"\n--- Verificación final antes del gráfico ---")
    print(f"Pilotos en order: {order}")
    for driver in order:
        driver_data = laps_df[laps_df['Driver'] == driver]
        print(f"Piloto {driver}: {len(driver_data)} vueltas")

    # Crear el violin plot
    sns.violinplot(
        data=laps_df,
        x="Driver",
        y="LapTimeSeconds",
        order=order,
        palette=palette_list,
        inner="quartile",
        cut=0,
        linewidth=1.0,
        ax=ax
    )

    # Puntos individuales encima (más legible)
    sns.stripplot(
        data=laps_df,
        x="Driver",
        y="LapTimeSeconds",
        order=order,
        color='yellow',
        size=3,
        jitter=True,
        alpha=0.6,
        ax=ax
    )

    # Formatear eje Y en mm:ss.s (ej: 1:12.34)
    def format_mmss(x, pos=None):
        if pd.isna(x) or x <= 0:
            return ""
        mins = int(x // 60)
        secs = x % 60
        return f"{mins}:{secs:05.2f}"

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_mmss))

    # CONFIGURACIÓN MEJORADA DE ETIQUETAS
    # Asegurar que las etiquetas del eje X sean visibles
    ax.set_xticks(range(len(order)))  # Forzar posiciones de ticks
    ax.set_xticklabels(order, rotation=45, ha='right')  # Etiquetas explícitas

    # Configurar colores para mejor contraste
    ax.tick_params(axis='x', labelsize=12, colors='black')
    ax.tick_params(axis='y', labelsize=10, colors='black')
    ax.set_xlabel("Piloto", color='black', fontsize=12, fontweight='bold')
    ax.set_ylabel("Tiempo de vuelta (mm:ss.ss)", color='black', fontsize=12, fontweight='bold')
    ax.set_title(f"Comparación de ritmo - {evento['EventName']} {year} - {sesion_tipo}",
                 color='black', fontsize=14, fontweight='bold')

    # Añadir grid para mejor legibilidad
    ax.grid(True, alpha=0.3)

    # Asegurar que todo sea visible
    plt.setp(ax.get_xticklabels(), visible=True)
    plt.setp(ax.get_yticklabels(), visible=True)

    # Ajustar layout y guardar
    plt.tight_layout()
    plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nGráfico guardado en: {fname}")

    # Mostrar información adicional
    print(f"\n--- Resumen estadístico ---")
    for driver in order:
        driver_times = laps_df[laps_df['Driver'] == driver]['LapTimeSeconds']
        if len(driver_times) > 0:
            avg_time = driver_times.mean()
            best_time = driver_times.min()
            print(f"{driver}: Mejor = {format_mmss(best_time)}, Media = {format_mmss(avg_time)}, Vueltas = {len(driver_times)}")

    plt.show()

# ----------------------------------------------------------------------------
# Pilot inividual
# ----------------------------------------------------------------------------
def accion_piloto_individual():
    """Ritmo de un piloto específico en una sesión"""
    # Cargar sesión usando la función común
    session, evento, year, sesion_tipo = cargar_sesion()

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

