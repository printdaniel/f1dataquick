import fastf1
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from datetime import datetime
import matplotlib.ticker as ticker
from datetime import timedelta
import constantes

# Definir ruta
CACHE_DIR = os.path.join(os.getcwd(), "cache")

# Crear si no existe
os.makedirs(CACHE_DIR, exist_ok=True)

# Habilitar caché
fastf1.Cache.enable_cache(CACHE_DIR)


team_colors_2025 = {
    "Red Bull": "#1E5BC6",  # Azul oscuro característico
    "Ferrari": "#ED1C24",   # Rosso Corsa (rojo Ferrari)
    "Mercedes": "#00D7B6",  # Plateado/cian con toques de negro
    "McLaren": "#FF8000",   # Naranja papaya
    "Aston Martin": "#2D8266",  # Verde británico
    "Alpine": "#0093D0",    # Azul claro de Renault
    "Williams": "#005AFF",  # Azul oscuro elegante
    "Kick Sauber": "#00FF00",  # Verde neón
    "Haas": "#B0B6B8",      # Gris plateado
    "Racing Bulls": "#F4F6F5"  # Blanco predominante
}
# ----------------------------------------------------------------------------
#  Funciones utilitarias (cambiar este nombre horrible)
# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------
# Comparación tiempos por vuelta
# ----------------------------------------------------------------------------

def accion_comparar_tiempos_vuelta():
    """Compara tiempos de vuelta entre pilotos en formato de tabla detallada."""

    # Cargar sesión (usando tu función existente)
    try:
        session, evento, year, sesion_tipo = cargar_sesion()
    except Exception as e:
        print(f"❌ Error al cargar sesión: {e}")
        return

    # Pedir pilotos a comparar
    while True:
        pilotos_input = input("\nIntroduce códigos de pilotos separados por coma (ej: VER,LEC,HAM): ")
        pilotos = [p.strip().upper() for p in pilotos_input.split(",") if p.strip()]

        if len(pilotos) >= 2:
            break
        else:
            print("⚠️ Debes ingresar al menos 2 pilotos.")

    print(f"\n📊 Cargando datos de {evento['EventName']} {year} - {sesion_tipo}...")

    # Cargar vueltas de todos los pilotos seleccionados
    try:
        laps = session.laps.pick_drivers(pilotos)

        if len(laps) == 0:
            print("❌ No se encontraron vueltas para los pilotos seleccionados")
            return

        laps_df = pd.DataFrame(laps)

    except Exception as e:
        print(f"❌ Error al cargar vueltas: {e}")
        return

    # Procesar tiempos de vuelta
    laps_df = procesar_tiempos_vuelta(laps_df)

    if laps_df.empty:
        print("❌ No hay datos válidos después del procesamiento")
        return

    # Mostrar diferentes opciones de visualización
    while True:
        print(f"\n🎯 OPCIONES DE COMPARACIÓN PARA {', '.join(pilotos)}:")
        print("1. Tabla completa de todas las vueltas")
        print("2. Solo vueltas rápidas (mejores tiempos)")
        print("3. Comparativa por stint (neumáticos)")
        print("4. Resumen estadístico completo")
        print("5. Volver al menú principal")

        opcion = input("\nSelecciona una opción (1-5): ").strip()

        if opcion == '1':
            mostrar_tabla_completa(laps_df, pilotos, evento, year, sesion_tipo)
        elif opcion == '2':
            mostrar_vueltas_rapidas(laps_df, pilotos, evento, year, sesion_tipo)
        elif opcion == '3':
            mostrar_comparativa_stints(laps_df, pilotos, evento, year, sesion_tipo)
        elif opcion == '4':
            mostrar_resumen_estadistico(laps_df, pilotos, evento, year, sesion_tipo)
        elif opcion == '5':
            print("Volviendo al menú principal...")
            break
        else:
            print("❌ Opción inválida. Intenta nuevamente.")

def procesar_tiempos_vuelta(laps_df):
    """Procesa y convierte los tiempos de vuelta a formato usable."""

    # Crear copia para no modificar el original
    df = laps_df.copy()

    # Convertir LapTime a segundos
    if 'LapTime' in df.columns:
        if pd.api.types.is_timedelta64_dtype(df['LapTime']):
            df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
        else:
            try:
                df['LapTime'] = pd.to_timedelta(df['LapTime'])
                df['LapTimeSeconds'] = df['LapTime'].dt.total_seconds()
            except:
                df['LapTimeSeconds'] = pd.to_numeric(df['LapTime'], errors='coerce')
    else:
        print("⚠️ No se encontró columna 'LapTime'")
        return pd.DataFrame()

    # Filtrar vueltas válidas (positivas y no nulas)
    df = df[df['LapTimeSeconds'] > 0]
    df = df.dropna(subset=['LapTimeSeconds', 'Driver', 'LapNumber'])

    return df

def formatear_tiempo(segundos):
    """Convierte segundos a formato mm:ss.sss"""
    if pd.isna(segundos) or segundos <= 0:
        return "N/A"
    mins = int(segundos // 60)
    secs = segundos % 60
    return f"{mins}:{secs:06.3f}"

def mostrar_tabla_completa(laps_df, pilotos, evento, year, sesion_tipo):
    """Muestra tabla completa con todas las vueltas de cada piloto."""

    print(f"\n📋 TABLA COMPLETA DE TIEMPOS - {evento['EventName']} {year} - {sesion_tipo}")
    print(f"Pilotos: {', '.join(pilotos)}")
    print("=" * 120)

    # Crear tabla pivote
    pivot_table = laps_df.pivot_table(
        index='LapNumber',
        columns='Driver',
        values='LapTimeSeconds',
        aggfunc='first'
    ).reset_index()

    # Reordenar columnas según orden de pilotos ingresado
    column_order = ['LapNumber'] + [p for p in pilotos if p in pivot_table.columns]
    pivot_table = pivot_table[column_order]

    # Renombrar columnas
    pivot_table = pivot_table.rename(columns={'LapNumber': 'Vuelta'})

    # Formatear tiempos
    for piloto in pilotos:
        if piloto in pivot_table.columns:
            pivot_table[piloto] = pivot_table[piloto].apply(formatear_tiempo)

    # Mostrar tabla
    print(f"{'Vuelta':<6} ", end="")
    for piloto in pilotos:
        if piloto in pivot_table.columns:
            print(f"{piloto:<12} ", end="")
    print()

    print("-" * 120)

    for _, fila in pivot_table.iterrows():
        print(f"{int(fila['Vuelta']):<6} ", end="")

        # Encontrar mejor tiempo de la vuelta
        mejores_tiempos = []
        for piloto in pilotos:
            if piloto in fila and fila[piloto] != 'N/A':
                try:
                    tiempo_str = fila[piloto]
                    mins, secs = tiempo_str.split(':')
                    tiempo_seg = float(mins) * 60 + float(secs)
                    mejores_tiempos.append((piloto, tiempo_seg))
                except:
                    continue

        mejor_piloto = min(mejores_tiempos, key=lambda x: x[1])[0] if mejores_tiempos else None

        for piloto in pilotos:
            if piloto in fila:
                tiempo = fila[piloto]
                if piloto == mejor_piloto and tiempo != 'N/A':
                    print(f"\033[92m{tiempo:<12}\033[0m ", end="")  # Verde
                else:
                    print(f"{tiempo:<12} ", end="")
            else:
                print(f"{'N/A':<12} ", end="")
        print()

    print("=" * 120)
    print("🎯 Leyenda: \033[92mVerde\033[0m = Mejor tiempo de la vuelta")

def mostrar_vueltas_rapidas(laps_df, pilotos, evento, year, sesion_tipo):
    """Muestra solo las mejores vueltas de cada piloto."""

    print(f"\n⚡ MEJORES TIEMPOS POR PILOTO - {evento['EventName']} {year} - {sesion_tipo}")
    print("=" * 80)

    resultados = []
    for piloto in pilotos:
        tiempos_piloto = laps_df[laps_df['Driver'] == piloto]['LapTimeSeconds']

        if len(tiempos_piloto) > 0:
            mejor_tiempo = tiempos_piloto.min()
            vuelta_mejor = laps_df[
                (laps_df['Driver'] == piloto) &
                (laps_df['LapTimeSeconds'] == mejor_tiempo)
            ]['LapNumber'].iloc[0]

            resultados.append({
                'Piloto': piloto,
                'Mejor Tiempo': mejor_tiempo,
                'Vuelta': vuelta_mejor,
                'Vueltas Válidas': len(tiempos_piloto)
            })

    # Ordenar por mejor tiempo
    resultados.sort(key=lambda x: x['Mejor Tiempo'])

    print(f"{'Pos':<4} {'Piloto':<8} {'Mejor Tiempo':<15} {'Vuelta':<8} {'Vueltas':<8}")
    print("-" * 80)

    for i, resultado in enumerate(resultados, 1):
        color_code = "\033[92m" if i == 1 else "\033[93m" if i == 2 else "\033[91m" if i == 3 else ""
        reset_code = "\033[0m" if i <= 3 else ""

        print(f"{color_code}{i:<4} {resultado['Piloto']:<8} {formatear_tiempo(resultado['Mejor Tiempo']):<15} {int(resultado['Vuelta']):<8} {resultado['Vueltas Válidas']:<8}{reset_code}")

def mostrar_comparativa_stints(laps_df, pilotos, evento, year, sesion_tipo):
    """Muestra comparativa organizada por stints (neumáticos)."""

    print(f"\n🔄 COMPARATIVA POR STINTS - {evento['EventName']} {year} - {sesion_tipo}")
    print("=" * 100)

    if 'Compound' not in laps_df.columns:
        print("ℹ️ No hay información de neumáticos disponible")
        return

    for piloto in pilotos:
        datos_piloto = laps_df[laps_df['Driver'] == piloto].copy()

        if len(datos_piloto) > 0:
            print(f"\n🏎️  PILOTO: {piloto}")
            print("-" * 60)

            # Agrupar por stint (usando cambios en Compound)
            datos_piloto = datos_piloto.sort_values('LapNumber')
            cambios_stint = datos_piloto['Compound'].ne(datos_piloto['Compound'].shift()).cumsum()

            for stint_num, stint_data in datos_piloto.groupby(cambios_stint):
                compound = stint_data['Compound'].iloc[0]
                vueltas_stint = stint_data['LapNumber'].tolist()
                mejor_tiempo = stint_data['LapTimeSeconds'].min()
                promedio_tiempo = stint_data['LapTimeSeconds'].mean()

                print(f"Stint {stint_num}: {compound} - Vueltas {min(vueltas_stint)}-{max(vueltas_stint)}")
                print(f"  Mejor: {formatear_tiempo(mejor_tiempo)} | Promedio: {formatear_tiempo(promedio_tiempo)}")
                print(f"  Vueltas: {len(vueltas_stint)}")
                print()

def mostrar_resumen_estadistico(laps_df, pilotos, evento, year, sesion_tipo):
    """Muestra resumen estadístico completo."""

    print(f"\n📈 RESUMEN ESTADÍSTICO - {evento['EventName']} {year} - {sesion_tipo}")
    print("=" * 100)

    estadisticas = []
    for piloto in pilotos:
        tiempos = laps_df[laps_df['Driver'] == piloto]['LapTimeSeconds']

        if len(tiempos) > 0:
            stats = {
                'Piloto': piloto,
                'Vueltas': len(tiempos),
                'Mejor': tiempos.min(),
                'Promedio': tiempos.mean(),
                'Mediana': tiempos.median(),
                'Std Dev': tiempos.std(),
                'Consistencia': tiempos.std() / tiempos.mean() * 100  # Coeficiente de variación
            }
            estadisticas.append(stats)

    # Ordenar por mejor tiempo
    estadisticas.sort(key=lambda x: x['Mejor'])

    print(f"{'Pos':<4} {'Piloto':<8} {'Mejor':<12} {'Promedio':<12} {'Mediana':<12} {'Consistencia':<12} {'Vueltas':<8}")
    print("-" * 100)

    for i, stats in enumerate(estadisticas, 1):
        color_code = "\033[92m" if i == 1 else "\033[93m" if i == 2 else "\033[91m" if i == 3 else ""
        reset_code = "\033[0m" if i <= 3 else ""

        consistencia = f"{stats['Consistencia']:.1f}%"

        print(f"{color_code}{i:<4} {stats['Piloto']:<8} "
              f"{formatear_tiempo(stats['Mejor']):<12} "
              f"{formatear_tiempo(stats['Promedio']):<12} "
              f"{formatear_tiempo(stats['Mediana']):<12} "
              f"{consistencia:<12} "
              f"{stats['Vueltas']:<8}{reset_code}")
# ----------------------------------------------------------------------------
# Rendimiento aerodinámico
# ----------------------------------------------------------------------------
def accion_eficiencia_aerodinamica_detallada():
    """Versión más detallada que usa datos específicos de la trampa de velocidad."""

    try:
        session, evento, year, sesion_tipo = cargar_sesion()
    except Exception as e:
        print(f"❌ Error al cargar sesión: {e}")
        return

    print(f"\n📊 Analizando eficiencia aerodinámica detallada...")

    try:
        laps = session.laps.pick_quicklaps()
        equipos = laps['Team'].unique()

        resultados_equipos = {}

        for equipo in equipos:
            print(f"🔍 Procesando {equipo}...")

            laps_equipo = laps[laps['Team'] == equipo]
            if len(laps_equipo) == 0:
                print(f"  ⚠️ No hay vueltas para {equipo}")
                continue

            # Tomar la vuelta más rápida del equipo
            vuelta_rapida = laps_equipo.loc[laps_equipo['LapTime'].idxmin()]
            telemetria = vuelta_rapida.get_telemetry()

            if telemetria is None or len(telemetria) == 0:
                print(f"  ⚠️ No hay telemetría para {equipo}")
                continue

            # Velocidad promedio de toda la vuelta
            velocidad_promedio = telemetria['Speed'].mean()

            # Velocidad máxima (en trampa de velocidad)
            velocidad_maxima = telemetria['Speed'].max()

            # Para la trampa de velocidad, usamos el último 10% de la vuelta (normalmente recta principal)
            # Esto es más robusto que depender de los datos del circuito
            ultimo_segmento = telemetria.tail(max(1, len(telemetria) // 10))
            velocidad_trampa = ultimo_segmento['Speed'].max()

            resultados_equipos[equipo] = {
                'velocidad_promedio': velocidad_promedio,
                'velocidad_maxima': velocidad_maxima,
                'velocidad_trampa': velocidad_trampa,
                'piloto': vuelta_rapida['Driver'],
                'vuelta_numero': vuelta_rapida['LapNumber'],
                'tiempo_vuelta': vuelta_rapida['LapTime']
            }

            print(f"  ✅ {equipo}: Vavg={velocidad_promedio:.1f} km/h, Vmax={velocidad_maxima:.1f} km/h, Vtrampa={velocidad_trampa:.1f} km/h")

        if resultados_equipos:
            crear_grafico_eficiencia_detallada(resultados_equipos, evento, year, sesion_tipo)
        else:
            print("❌ No se pudieron procesar datos para ningún equipo")

    except Exception as e:
        print(f"❌ Error en análisis detallado: {e}")
        import traceback
        traceback.print_exc()

def crear_grafico_eficiencia_detallada(resultados_equipos, evento, year, sesion_tipo):
    """Crea el gráfico de eficiencia aerodinámica detallado."""

    import matplotlib.pyplot as plt
    import numpy as np

    # Configuración del estilo
    plt.style.use('default')
    sns.set_theme(style="whitegrid")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Preparar datos
    equipos = list(resultados_equipos.keys())
    v_promedio = [resultados_equipos[eq]['velocidad_promedio'] for eq in equipos]
    v_maxima = [resultados_equipos[eq]['velocidad_maxima'] for eq in equipos]
    v_trampa = [resultados_equipos[eq]['velocidad_trampa'] for eq in equipos]

   # GRÁFICO 1: Velocidad Máxima vs Velocidad Promedio

    for i, equipo in enumerate(equipos):
        color = team_colors_2025.get(equipo, '#888888')
        ax1.scatter(v_promedio[i], v_maxima[i], c=color, s=150, alpha=0.8,
                   edgecolors='black', linewidth=1.5, label=equipo)

    # Línea de tendencia
    if len(v_promedio) > 1:
        z = np.polyfit(v_promedio, v_maxima, 1)
        p = np.poly1d(z)
        ax1.plot(v_promedio, p(v_promedio), "r--", alpha=0.7, linewidth=2,
                label=f'Tendencia (pendiente: {z[0]:.2f})')

    ax1.set_xlabel('Velocidad Promedio (km/h)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Velocidad Máxima (km/h)', fontsize=12, fontweight='bold')
    ax1.set_title('Eficiencia General: Vmax vs Vpromedio', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)

    # GRÁFICO 2: Velocidad en Trampa vs Velocidad Promedio
    for i, equipo in enumerate(equipos):
        color = team_colors_2025.get(equipo, '#888888')
        ax2.scatter(v_promedio[i], v_trampa[i], c=color, s=150, alpha=0.8,
                   edgecolors='black', linewidth=1.5, label=equipo)

    # Línea de tendencia
    if len(v_promedio) > 1:
        z2 = np.polyfit(v_promedio, v_trampa, 1)
        p2 = np.poly1d(z2)
        ax2.plot(v_promedio, p2(v_promedio), "r--", alpha=0.7, linewidth=2,
                label=f'Tendencia (pendiente: {z2[0]:.2f})')

    ax2.set_xlabel('Velocidad Promedio (km/h)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Velocidad en Trampa (km/h)', fontsize=12, fontweight='bold')
    ax2.set_title('Eficiencia en Recta: Vtrampa vs Vpromedio', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best', fontsize=9)

    # Título general
    fig.suptitle(f'Eficiencia Aerodinámica - {evento["EventName"]} {year} - {sesion_tipo}',
                 fontsize=16, fontweight='bold', y=0.98)

    # Mostrar tabla de datos
    print(f"\n📋 DATOS DETALLADOS DE EFICIENCIA AERODINÁMICA:")
    print("="*100)
    print(f"{'Equipo':<15} {'Piloto':<8} {'Vuelta':<6} {'Vavg':<8} {'Vmax':<8} {'Vtrampa':<8} {'Diff Vmax-Vavg':<12} {'Eficiencia':<10}")
    print("-"*100)

    for equipo in sorted(equipos, key=lambda x: resultados_equipos[x]['velocidad_promedio'], reverse=True):
        datos = resultados_equipos[equipo]
        diff_vmax_vavg = datos['velocidad_maxima'] - datos['velocidad_promedio']
        eficiencia = (datos['velocidad_trampa'] / datos['velocidad_promedio'] - 1) * 100

        print(f"{equipo:<15} {datos['piloto']:<8} {datos['vuelta_numero']:<6} "
              f"{datos['velocidad_promedio']:<8.1f} {datos['velocidad_maxima']:<8.1f} "
              f"{datos['velocidad_trampa']:<8.1f} {diff_vmax_vavg:<12.1f} {eficiencia:<10.1f}%")

    # Análisis interpretativo
    print(f"\n💡 INTERPRETACIÓN:")
    print("• Alta Vpromedio + Alta Vmax: Excelente eficiencia aerodinámica")
    print("• Alta Vmax + Baja Vpromedio: Buen motor, mala downforce en curvas")
    print("• Baja Vmax + Alta Vpromedio: Buen downforce, motor limitado")
    print("• Alta diferencia Vtrampa-Vpromedio: Buen rendimiento en rectas")

    # Guardar gráfico
    out_dir = "output/figures"
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{out_dir}/eficiencia_aerodinamica_detallada_{evento['EventName'].replace(' ','_')}_{year}_{sesion_tipo}.png"

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n💾 Gráfico guardado en: {filename}")

    plt.show()

# Y también corregimos la función principal para que sea más robusta
def accion_eficiencia_aerodinamica():
    """Analiza la eficiencia aerodinámica: Velocidad máxima vs Velocidad promedio por equipo."""

    try:
        session, evento, year, sesion_tipo = cargar_sesion()
    except Exception as e:
        print(f"❌ Error al cargar sesión: {e}")
        return

    print(f"\n📊 Analizando eficiencia aerodinámica - {evento['EventName']} {year} - {sesion_tipo}...")

    try:
        # Obtener todas las vueltas y filtrar las rápidas
        all_laps = session.laps
        laps = all_laps.pick_quicklaps()

        if len(laps) == 0:
            print("⚠️ No se encontraron vueltas rápidas, usando todas las vueltas...")
            laps = all_laps

        # Obtener equipos únicos
        equipos = laps['Team'].dropna().unique()
        print(f"🏁 Equipos encontrados: {', '.join(equipos)}")

        if len(equipos) == 0:
            print("❌ No se encontraron equipos en los datos")
            return

    except Exception as e:
        print(f"❌ Error al cargar datos: {e}")
        import traceback
        traceback.print_exc()
        return

    # Diccionario para almacenar resultados por equipo
    resultados_equipos = {}

    for equipo in equipos:
        try:
            print(f"🔍 Procesando {equipo}...")

            # Filtrar vueltas del equipo
            laps_equipo = laps[laps['Team'] == equipo]

            if len(laps_equipo) == 0:
                print(f"  ⚠️ No hay vueltas para {equipo}")
                continue

            # Encontrar la vuelta más rápida del equipo
            vuelta_rapida_equipo = laps_equipo.loc[laps_equipo['LapTime'].idxmin()]

            # Obtener telemetría de esa vuelta
            telemetria = vuelta_rapida_equipo.get_telemetry()

            if telemetria is None or len(telemetria) == 0:
                print(f"  ⚠️ No hay telemetría para {equipo}")
                continue

            # Calcular velocidades
            velocidad_maxima = telemetria['Speed'].max()
            velocidad_promedio = telemetria['Speed'].mean()

            # Almacenar resultados
            resultados_equipos[equipo] = {
                'velocidad_maxima': velocidad_maxima,
                'velocidad_promedio': velocidad_promedio,
                'piloto': vuelta_rapida_equipo['Driver'],
                'tiempo_vuelta': vuelta_rapida_equipo['LapTime'],
                'vuelta_numero': vuelta_rapida_equipo['LapNumber']
            }

            print(f"  ✅ {equipo}: Vmax={velocidad_maxima:.1f} km/h, Vavg={velocidad_promedio:.1f} km/h")

        except Exception as e:
            print(f"  ❌ Error procesando {equipo}: {e}")
            continue

    if not resultados_equipos:
        print("❌ No se pudieron procesar datos para ningún equipo")
        return

    # Crear el gráfico
    crear_grafico_eficiencia_aerodinamica(resultados_equipos, evento, year, sesion_tipo)


def salir():
    print("👋 Saliendo del programa... Hasta la próxima!")

def menu_principal():
    while True:
        print("\n--- Menú Principal ---")
        print("[ 1 ]. Comparar ritmo entre pilotos")
        print("[ 2 ]. Ritmo de un piloto específico")
        print("[ 3 ]. Tiempos de vuelta")
        print("[ 4 ]. Eficiencia Aerodinámica")
        print("[ 5 ]. Salir")

        opcion = input("Elige una opción (1, 2, 3): ")

        if opcion == '1':
            accion_comparar_pilotos()
        elif opcion == '2':
            accion_piloto_individual()
        elif opcion == '3':
            accion_comparar_tiempos_vuelta()  # Nueva función de tabla
        elif opcion == '4':
            accion_eficiencia_aerodinamica_detallada()
        elif opcion == '5':
            salir()
            break
        else:
            print("❌ Opción no válida. Por favor, elige un número del 1 al 3.")

if __name__ == "__main__":
    menu_principal()
