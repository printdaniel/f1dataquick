# main.py - Análisis de Datos de Fórmula 1
from constantes import *
import fastf1
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import matplotlib.ticker as ticker
from datetime import timedelta
from datetime import datetime
import numpy as np
import logging
from functools import lru_cache

# Configuración de logging
def setup_logging():
    """Configura el sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('f1_analytics.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# Definir ruta
CACHE_DIR = os.path.join(os.getcwd(), "cache")

# Crear si no existe
os.makedirs(CACHE_DIR, exist_ok=True)

# Habilitar caché
fastf1.Cache.enable_cache(CACHE_DIR)

# ----------------------------------------------------------------------------
#  Funciones utilitarias
# ----------------------------------------------------------------------------
def elegir_gp(year: int):
    """Muestra el calendario de un año y permite elegir GP"""
    try:
        schedule = fastf1.get_event_schedule(year)
        print(f"\n--- Calendario {year} ---")
        for idx, row in schedule.iterrows():
            print(f"[ {row['RoundNumber']:2d} ] {row['EventName']} - {row['EventDate'].date()}")

        while True:
            try:
                ronda = int(input("Elige el número de ronda (ej: 1, 2, 3...): "))
                if ronda in schedule["RoundNumber"].values:
                    evento = schedule.loc[schedule["RoundNumber"] == ronda].iloc[0]
                    logger.info(f"Evento seleccionado: {evento['EventName']} - Ronda {ronda}")
                    return evento
                else:
                    print("❌ Ronda inválida, intenta de nuevo.")
            except ValueError:
                print("❌ Ingresa un número válido.")
    except Exception as e:
        logger.error(f"Error al obtener calendario: {e}")
        raise

def elegir_sesion(evento):
    """Permite elegir el tipo de sesión"""
    print("\n--- Sesiones disponibles ---")
    for k, v in sesiones_validas.items():
        print(f"{k:4s} → {v}")

    while True:
        sesion = input("Elige sesión (FP1, FP2, FP3, Q, R): ").upper()
        if sesion in sesiones_validas:
            logger.info(f"Sesión seleccionada: {sesiones_validas[sesion]}")
            return sesiones_validas[sesion]
        else:
            print("❌ Sesión inválida. Intenta de nuevo.")

@lru_cache(maxsize=32)
def cargar_sesion_cached(year, ronda, sesion_tipo):
    """Carga una sesión con cache para mejor rendimiento"""
    try:
        session = fastf1.get_session(year, ronda, sesion_tipo)
        session.load()
        logger.info(f"Sesión cargada: {year} Ronda {ronda} - {sesion_tipo}")
        return session
    except Exception as e:
        logger.error(f"Error cargando sesión: {e}")
        raise

def cargar_sesion():
    """Solicita año, evento y tipo de sesión, y carga una sesión de FastF1."""
    try:
        year = int(input("Año de la temporada (ej: 2025): "))
        evento = elegir_gp(year)
        sesion_tipo = elegir_sesion(evento)
        print(f"\nCargando datos: {evento['EventName']} {year} - {sesion_tipo}...")
        session = cargar_sesion_cached(year, int(evento["RoundNumber"]), sesion_tipo)
        return session, evento, year, sesion_tipo
    except Exception as e:
        logger.error(f"Error en cargar_sesion: {e}")
        raise

def verificar_datos_sesion(session):
    """Verificación comprehensiva de datos disponibles"""
    checks = {
        'laps': len(session.laps) > 0,
        'drivers': len(session.drivers) > 0,
        'telemetry': False,
        'weather': hasattr(session, 'weather_data') and session.weather_data is not None
    }

    # Verificar telemetría
    if checks['laps']:
        try:
            sample_lap = session.laps.iloc[0]
            telemetry = sample_lap.get_telemetry()
            checks['telemetry'] = len(telemetry) > 0 if telemetry is not None else False
        except Exception as e:
            logger.warning(f"No se pudo verificar telemetría: {e}")
            checks['telemetry'] = False

    logger.info(f"Verificación de datos: {checks}")
    return checks

# ----------------------------------------------------------------------------
# Comparación entre pilotos - CORREGIDO
# ----------------------------------------------------------------------------
def accion_comparar_pilotos():
    """Comparar ritmo entre pilotos en una sesión con violin plot (robusto)."""
    try:
        session, evento, year, sesion_tipo = cargar_sesion()

        # Verificar datos disponibles
        checks = verificar_datos_sesion(session)
        if not checks['laps']:
            print("❌ No hay datos de vueltas disponibles para esta sesión")
            return

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

        # Convertir a DataFrame por seguridad
        laps_df = pd.DataFrame(laps)

        if laps_df.empty:
            print("❌ No se encontraron vueltas válidas para los pilotos seleccionados")
            return

        # --- Diagnóstico rápido
        print("\n--- Diagnóstico de columnas y tipos ---")
        print(laps_df.dtypes)
        print("Pilotos encontrados:", sorted(laps_df['Driver'].unique().tolist()))
        print("Primeras filas:")
        print(laps_df[['Driver','LapTime','LapNumber']].head())

        # Asegurar LapTimeSeconds (float)
        if 'LapTime' not in laps_df.columns:
            raise RuntimeError("No se encontró la columna 'LapTime' en los datos.")

        # Procesar tiempos de vuelta
        if pd.api.types.is_timedelta64_dtype(laps_df['LapTime']):
            laps_df['LapTimeSeconds'] = laps_df['LapTime'].dt.total_seconds()
        else:
            try:
                laps_df['LapTime'] = pd.to_timedelta(laps_df['LapTime'])
                laps_df['LapTimeSeconds'] = laps_df['LapTime'].dt.total_seconds()
            except Exception:
                laps_df['LapTimeSeconds'] = pd.to_numeric(laps_df['LapTime'], errors='coerce')

        # Quitar filas sin tiempo
        laps_df = laps_df.dropna(subset=['LapTimeSeconds'])
        laps_df = laps_df[laps_df['LapTimeSeconds'] > 0]

        if laps_df.empty:
            raise RuntimeError("No quedan vueltas con tiempo válido después del filtrado.")

        # Asegurar que la columna Driver sea string y los códigos estén en mayúsculas
        laps_df['Driver'] = laps_df['Driver'].astype(str).str.upper()

        # Forzar orden de pilotos
        present_drivers = [d for d in pilotos if d in laps_df['Driver'].unique()]
        if not present_drivers:
            present_drivers = sorted(laps_df['Driver'].unique())
        order = present_drivers

        # Preparar paleta
        palette_list = [driver_colors.get(d, "#888888") for d in order]

        # Crear carpeta para figuras
        out_dir = "output/figures"
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{out_dir}/violin_comparacion_{evento['EventName'].replace(' ','_')}_{year}_{sesion_tipo}.png"

        # Estilo y tema
        plt.style.use("default")
        sns.set_theme(style="darkgrid")

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

        # CORREGIDO: Puntos individuales encima (versión funcional)
        sns.stripplot(
            data=laps_df,
            x="Driver",
            y="LapTimeSeconds",
            order=order,
            color='yellow',
            size=3,
            jitter=True,
            alpha=0.7,
            ax=ax,
            legend=False
        )

        # Formatear eje Y en mm:ss.s (ej: 1:12.34)
        def format_mmss(x, pos=None):
            if pd.isna(x) or x <= 0:
                return ""
            mins = int(x // 60)
            secs = x % 60
            return f"{mins}:{secs:05.2f}"

        ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_mmss))

        # Configuración mejorada de etiquetas
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=45, ha='right')

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

    except Exception as e:
        logger.error(f"Error en comparación de pilotos: {e}")
        print(f"❌ Error: {e}")

# ----------------------------------------------------------------------------
# Piloto individual
# ----------------------------------------------------------------------------
def accion_piloto_individual():
    """Ritmo de un piloto específico en una sesión con análisis de compuestos"""
    try:
        session, evento, year, sesion_tipo = cargar_sesion()

        # Verificar datos disponibles
        checks = verificar_datos_sesion(session)
        if not checks['laps']:
            print("❌ No hay datos de vueltas disponibles para esta sesión")
            return

        piloto = input("Código de piloto (ej: VER, HAM, ALO): ").upper()

        # Obtener todas las vueltas del piloto (no solo las rápidas)
        laps = session.laps.pick_driver(piloto)

        if laps.empty:
            print(f"❌ No se encontraron vueltas para el piloto {piloto}")
            return

        # Convertir a DataFrame para mayor control
        laps_df = pd.DataFrame(laps)

        # Procesar tiempos de vuelta
        if 'LapTime' in laps_df.columns and pd.api.types.is_timedelta64_dtype(laps_df['LapTime']):
            laps_df['LapTimeSeconds'] = laps_df['LapTime'].dt.total_seconds()
        else:
            print("⚠️ No se pudieron procesar los tiempos de vuelta")
            return

        # Filtrar vueltas válidas
        laps_df = laps_df.dropna(subset=['LapTimeSeconds'])
        laps_df = laps_df[laps_df['LapTimeSeconds'] > 0]

        if laps_df.empty:
            print(f"❌ No hay vueltas válidas para el piloto {piloto}")
            return

        # Crear el gráfico
        plt.style.use('default')
        sns.set_theme(style="whitegrid")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # GRÁFICO 1: Violin plot con puntos por compuesto
        tiempos_totales = laps_df['LapTimeSeconds'].dropna()

        # Crear violin plot base
        violin_parts = ax1.violinplot([tiempos_totales], showmeans=True, showmedians=True)

        # Personalizar el violin plot
        for pc in violin_parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.6)

        violin_parts['cmeans'].set_color('red')
        violin_parts['cmedians'].set_color('black')

        # Añadir puntos individuales coloreados por compuesto
        for idx, lap in laps_df.iterrows():
            compound = lap['Compound'] if pd.notna(lap['Compound']) else 'UNKNOWN'
            color = compound_colors.get(compound, 'gray')

            # Jitter para evitar superposición de puntos
            jitter = np.random.normal(0, 0.02)

            ax1.scatter(1 + jitter, lap['LapTimeSeconds'],
                       c=color, s=50, alpha=0.7, edgecolors='black', linewidth=0.5,
                       label=compound if compound not in [l.get_label() for l in ax1.collections] else "")

        ax1.set_xlabel('Distribución', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Tiempo de Vuelta (segundos)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Distribución de Tiempos - {piloto}', fontsize=14, fontweight='bold')
        ax1.set_xticks([1])
        ax1.set_xticklabels([f'{piloto}\n(n={len(tiempos_totales)} vueltas)'])

        # Formatear eje Y en mm:ss
        def format_segundos(x, pos=None):
            if pd.isna(x) or x <= 0:
                return ""
            mins = int(x // 60)
            secs = x % 60
            return f"{mins}:{secs:05.2f}"

        ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_segundos))

        # GRÁFICO 2: Evolución de tiempos por vuelta con compuestos
        laps_df_sorted = laps_df.sort_values('LapNumber')

        # Scatter plot por número de vuelta
        for idx, lap in laps_df_sorted.iterrows():
            compound = lap['Compound'] if pd.notna(lap['Compound']) else 'UNKNOWN'
            color = compound_colors.get(compound, 'gray')

            ax2.scatter(lap['LapNumber'], lap['LapTimeSeconds'],
                       c=color, s=60, alpha=0.8, edgecolors='black', linewidth=0.8,
                       label=compound if compound not in [l.get_label() for l in ax2.collections] else "")

        # Línea que conecta los puntos
        ax2.plot(laps_df_sorted['LapNumber'], laps_df_sorted['LapTimeSeconds'],
                'gray', alpha=0.3, linewidth=1)

        ax2.set_xlabel('Número de Vuelta', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Tiempo de Vuelta', fontsize=12, fontweight='bold')
        ax2.set_title(f'Evolución de Tiempos - {piloto}', fontsize=14, fontweight='bold')
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_segundos))
        ax2.grid(True, alpha=0.3)

        # Leyenda unificada para compuestos
        handles_labels = {}
        for ax in [ax1, ax2]:
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in handles_labels:
                    handles_labels[label] = handle

        # Crear leyenda única
        if handles_labels:
            fig.legend(handles_labels.values(), handles_labels.keys(),
                      title="Compuestos", loc='upper center',
                      bbox_to_anchor=(0.5, 0.05), ncol=len(handles_labels))

        # Título general
        fig.suptitle(f'Análisis de Ritmo - {piloto} - {evento["EventName"]} {year} - {sesion_tipo}',
                     fontsize=16, fontweight='bold', y=0.98)

        # Estadísticas resumen
        print(f"\n📊 ESTADÍSTICAS DE {piloto}:")
        print("="*50)
        print(f"Vueltas totales: {len(laps_df)}")
        print(f"Mejor tiempo: {format_segundos(laps_df['LapTimeSeconds'].min())}")
        print(f"Tiempo promedio: {format_segundos(laps_df['LapTimeSeconds'].mean())}")
        print(f"Consistencia (std): {laps_df['LapTimeSeconds'].std():.2f} segundos")

        # Análisis por compuestos
        if 'Compound' in laps_df.columns:
            print(f"\n🏁 ANÁLISIS POR COMPUESTOS:")
            compounds_used = laps_df['Compound'].value_counts()
            for compound, count in compounds_used.items():
                if pd.notna(compound):
                    compound_times = laps_df[laps_df['Compound'] == compound]['LapTimeSeconds']
                    if len(compound_times) > 0:
                        print(f"  {compound}: {count} vueltas | Mejor: {format_segundos(compound_times.min())} | Promedio: {format_segundos(compound_times.mean())}")

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Espacio para la leyenda

        # Opcional: Guardar gráfico
        guardar = input("¿Guardar gráfico? (s/n): ").lower()
        if guardar == 's':
            out_dir = "output/figures"
            os.makedirs(out_dir, exist_ok=True)
            filename = f"{out_dir}/ritmo_individual_{piloto}_{evento['EventName'].replace(' ','_')}_{year}_{sesion_tipo}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"💾 Gráfico guardado en: {filename}")

        plt.show()

    except Exception as e:
        logger.error(f"Error en análisis individual: {e}")
        print(f"❌ Error: {e}")

# ----------------------------------------------------------------------------
# Comparación tiempos por vuelta
# ----------------------------------------------------------------------------
def accion_comparar_tiempos_vuelta():
    """Compara tiempos de vuelta entre pilotos en formato de tabla detallada."""
    try:
        session, evento, year, sesion_tipo = cargar_sesion()

        # Verificar datos disponibles
        checks = verificar_datos_sesion(session)
        if not checks['laps']:
            print("❌ No hay datos de vueltas disponibles para esta sesión")
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
        laps = session.laps.pick_drivers(pilotos)

        if len(laps) == 0:
            print("❌ No se encontraron vueltas para los pilotos seleccionados")
            return

        laps_df = pd.DataFrame(laps)
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

    except Exception as e:
        logger.error(f"Error en comparación de tiempos: {e}")
        print(f"❌ Error: {e}")

def procesar_tiempos_vuelta(laps_df):
    """Procesa y convierte los tiempos de vuelta a formato usable."""
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
# Eficiencia aerodinámica en recta
# ----------------------------------------------------------------------------
def accion_eficiencia_aerodinamica_detallada():
    """Versión más detallada que usa datos específicos de la trampa de velocidad."""
    try:
        session, evento, year, sesion_tipo = cargar_sesion()

        # Verificar datos disponibles
        checks = verificar_datos_sesion(session)
        if not checks['laps']:
            print("❌ No hay datos de vueltas disponibles para esta sesión")
            return
        if not checks['telemetry']:
            print("❌ No hay datos de telemetría disponibles para esta sesión")
            return

        print(f"\n📊 Analizando eficiencia aerodinámica detallada...")

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
            crear_grafico_eficiencia_recta(resultados_equipos, evento, year, sesion_tipo)
        else:
            print("❌ No se pudieron procesar datos para ningún equipo")

    except Exception as e:
        logger.error(f"Error en análisis de eficiencia: {e}")
        print(f"❌ Error: {e}")

def crear_grafico_eficiencia_recta(resultados_equipos, evento, year, sesion_tipo):
    """Crea el gráfico de eficiencia en recta: Velocidad en Trampa vs Velocidad Promedio."""
    # Configuración del estilo
    plt.style.use('default')
    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(figsize=(14, 10))

    # Preparar datos
    equipos = list(resultados_equipos.keys())
    v_promedio = [resultados_equipos[eq]['velocidad_promedio'] for eq in equipos]
    v_trampa = [resultados_equipos[eq]['velocidad_trampa'] for eq in equipos]

    # Crear scatter plot con etiquetas en los puntos
    for i, equipo in enumerate(equipos):
        color = team_colors_2025.get(equipo, '#888888')

        # Punto principal
        ax.scatter(v_promedio[i], v_trampa[i], c=color, s=200, alpha=0.8,
                   edgecolors='black', linewidth=2)

        # Etiqueta con nombre del equipo - SIN recuadro de anotaciones
        ax.annotate(equipo,
                   (v_promedio[i], v_trampa[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   alpha=0.9)

    # Línea de tendencia
    if len(v_promedio) > 1:
        z = np.polyfit(v_promedio, v_trampa, 1)
        p = np.poly1d(z)
        ax.plot(v_promedio, p(v_promedio), "r--", alpha=0.7, linewidth=2,
                label=f'Tendencia (pendiente: {z[0]:.2f})')

    # Configurar ejes y título
    ax.set_xlabel('Velocidad Promedio (km/h)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Velocidad en Trampa de Velocidad (km/h)', fontsize=14, fontweight='bold')

    titulo = f"Eficiencia en Recta - {evento['EventName']} {year} - {sesion_tipo}"
    ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)

    # Cuadrícula
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Leyenda de tendencia
    if len(v_promedio) > 1:
        ax.legend(loc='best')

    # Ajustar límites de ejes
    margen_x = (max(v_promedio) - min(v_promedio)) * 0.1
    margen_y = (max(v_trampa) - min(v_trampa)) * 0.1
    ax.set_xlim(min(v_promedio) - margen_x, max(v_promedio) + margen_x)
    ax.set_ylim(min(v_trampa) - margen_y, max(v_trampa) + margen_y)

    # Mostrar tabla de datos
    print(f"\n📋 DATOS DE EFICIENCIA EN RECTA:")
    print("="*90)
    print(f"{'Equipo':<15} {'Piloto':<8} {'Vuelta':<6} {'Vavg':<8} {'Vtrampa':<10} {'Diferencia':<12} {'Eficiencia':<12}")
    print("-"*90)

    for equipo in sorted(equipos, key=lambda x: resultados_equipos[x]['velocidad_promedio'], reverse=True):
        datos = resultados_equipos[equipo]
        diferencia = datos['velocidad_trampa'] - datos['velocidad_promedio']
        eficiencia = (datos['velocidad_trampa'] / datos['velocidad_promedio'] - 1) * 100

        print(f"{equipo:<15} {datos['piloto']:<8} {datos['vuelta_numero']:<6} "
              f"{datos['velocidad_promedio']:<8.1f} {datos['velocidad_trampa']:<10.1f} "
              f"{diferencia:<12.1f} {eficiencia:<12.1f}%")

    # Análisis interpretativo mejorado
    print(f"\n💡 INTERPRETACIÓN DEL GRÁFICO:")
    print("• 📈 ALTA Vpromedio + ALTA Vtrampa: Excelente eficiencia aerodinámica (equipo completo)")
    print("• 🚀 BAJA Vpromedio + ALTA Vtrampa: Buen motor/eficiencia en rectas, pero mala downforce en curvas")
    print("• 🏎️  ALTA Vpromedio + BAJA Vtrampa: Buen downforce en curvas, pero motor limitado en rectas")
    print("• 📉 BAJA Vpromedio + BAJA Vtrampa: Problemas generales de rendimiento")
    print(f"• 📊 Diferencia Vtrampa-Vpromedio: Indica la ganancia específica en rectas")

    # Calcular y mostrar rankings
    print(f"\n🏆 RANKING POR EFICIENCIA EN RECTA:")
    print("="*60)

    # Ranking por eficiencia (% ganancia en rectas)
    ranking_eficiencia = sorted(equipos,
                               key=lambda x: (resultados_equipos[x]['velocidad_trampa'] / resultados_equipos[x]['velocidad_promedio'] - 1) * 100,
                               reverse=True)

    print(f"{'Pos':<4} {'Equipo':<15} {'Eficiencia':<12} {'Vtrampa':<10}")
    print("-"*60)
    for i, equipo in enumerate(ranking_eficiencia, 1):
        datos = resultados_equipos[equipo]
        eficiencia = (datos['velocidad_trampa'] / datos['velocidad_promedio'] - 1) * 100
        color_code = "\033[92m" if i == 1 else "\033[93m" if i == 2 else "\033[91m" if i == 3 else ""
        reset_code = "\033[0m" if i <= 3 else ""

        print(f"{color_code}{i:<4} {equipo:<15} {eficiencia:<11.1f}% {datos['velocidad_trampa']:<9.1f}{reset_code}")

    # Guardar gráfico
    out_dir = "output/figures"
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{out_dir}/eficiencia_recta_{evento['EventName'].replace(' ','_')}_{year}_{sesion_tipo}.png"

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n💾 Gráfico guardado en: {filename}")

    plt.show()

# ==========================================================================
# Eficiencia aerodinámica General
# ==========================================================================
def accion_eficiencia_general():
    """Analiza la eficiencia general: Velocidad Máxima vs Velocidad Promedio"""
    try:
        session, evento, year, sesion_tipo = cargar_sesion()

        # Verificar datos disponibles
        checks = verificar_datos_sesion(session)
        if not checks['laps']:
            print("❌ No hay datos de vueltas disponibles para esta sesión")
            return
        if not checks['telemetry']:
            print("❌ No hay datos de telemetría disponibles para esta sesión")
            return

        print(f"\n📊 Analizando eficiencia general...")

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

            # Velocidad máxima (en cualquier punto del circuito)
            velocidad_maxima = telemetria['Speed'].max()

            resultados_equipos[equipo] = {
                'velocidad_promedio': velocidad_promedio,
                'velocidad_maxima': velocidad_maxima,
                'piloto': vuelta_rapida['Driver'],
                'vuelta_numero': vuelta_rapida['LapNumber'],
                'tiempo_vuelta': vuelta_rapida['LapTime']
            }

            print(f"  ✅ {equipo}: Vavg={velocidad_promedio:.1f}, Vmax={velocidad_maxima:.1f} km/h")

        if resultados_equipos:
            crear_grafico_eficiencia_general(resultados_equipos, evento, year, sesion_tipo)
        else:
            print("❌ No se pudieron procesar datos para ningún equipo")

    except Exception as e:
        logger.error(f"Error en análisis general: {e}")
        print(f"❌ Error: {e}")

def crear_grafico_eficiencia_general(resultados_equipos, evento, year, sesion_tipo):
    """Crea el gráfico de eficiencia general: Velocidad Máxima vs Velocidad Promedio."""
    # Configuración del estilo
    plt.style.use('default')
    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots(figsize=(14, 10))

    # Preparar datos
    equipos = list(resultados_equipos.keys())
    v_promedio = [resultados_equipos[eq]['velocidad_promedio'] for eq in equipos]
    v_maxima = [resultados_equipos[eq]['velocidad_maxima'] for eq in equipos]

    # Crear scatter plot con etiquetas en los puntos
    for i, equipo in enumerate(equipos):
        color = team_colors_2025.get(equipo, '#888888')

        # Punto principal
        ax.scatter(v_promedio[i], v_maxima[i], c=color, s=200, alpha=0.8,
                   edgecolors='black', linewidth=2)

        # Etiqueta con nombre del equipo
        ax.annotate(equipo,
                   (v_promedio[i], v_maxima[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   alpha=0.9)

    # Línea de tendencia
    if len(v_promedio) > 1:
        z = np.polyfit(v_promedio, v_maxima, 1)
        p = np.poly1d(z)
        ax.plot(v_promedio, p(v_promedio), "r--", alpha=0.7, linewidth=2,
                label=f'Tendencia (pendiente: {z[0]:.2f})')

    # Configurar ejes y título
    ax.set_xlabel('Velocidad Promedio (km/h)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Velocidad Máxima (km/h)', fontsize=14, fontweight='bold')

    titulo = f"Eficiencia General - {evento['EventName']} {year} - {sesion_tipo}"
    ax.set_title(titulo, fontsize=16, fontweight='bold', pad=20)

    # Cuadrícula
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Leyenda de tendencia
    if len(v_promedio) > 1:
        ax.legend(loc='best')

    # Ajustar límites de ejes
    margen_x = (max(v_promedio) - min(v_promedio)) * 0.1
    margen_y = (max(v_maxima) - min(v_maxima)) * 0.1
    ax.set_xlim(min(v_promedio) - margen_x, max(v_promedio) + margen_x)
    ax.set_ylim(min(v_maxima) - margen_y, max(v_maxima) + margen_y)

    # Mostrar tabla de datos
    print(f"\n📋 DATOS DE EFICIENCIA GENERAL:")
    print("="*90)
    print(f"{'Equipo':<15} {'Piloto':<8} {'Vuelta':<6} {'Vavg':<8} {'Vmax':<10} {'Diferencia':<12} {'Eficiencia':<12}")
    print("-"*90)

    for equipo in sorted(equipos, key=lambda x: resultados_equipos[x]['velocidad_promedio'], reverse=True):
        datos = resultados_equipos[equipo]
        diferencia = datos['velocidad_maxima'] - datos['velocidad_promedio']
        eficiencia = (datos['velocidad_maxima'] / datos['velocidad_promedio'] - 1) * 100

        print(f"{equipo:<15} {datos['piloto']:<8} {datos['vuelta_numero']:<6} "
              f"{datos['velocidad_promedio']:<8.1f} {datos['velocidad_maxima']:<10.1f} "
              f"{diferencia:<12.1f} {eficiencia:<12.1f}%")

    # Análisis interpretativo
    print(f"\n💡 INTERPRETACIÓN DEL GRÁFICO:")
    print("• 📈 ALTA Vpromedio + ALTA Vmax: Excelente eficiencia general (equipo completo)")
    print("• 🚀 BAJA Vpromedio + ALTA Vmax: Gran potencia motor, pero mala downforce/aerodinámica")
    print("• 🏎️  ALTA Vpromedio + BAJA Vmax: Buena aerodinámica, pero motor limitado")
    print("• 📉 BAJA Vpromedio + BAJA Vmax: Problemas generales de rendimiento")
    print(f"• 📊 Diferencia Vmax-Vpromedio: Indica el potencial máximo del motor")

    # Calcular y mostrar rankings
    print(f"\n🏆 RANKING POR EFICIENCIA GENERAL:")
    print("="*60)

    # Ranking por eficiencia (% ganancia máxima)
    ranking_eficiencia = sorted(equipos,
                               key=lambda x: (resultados_equipos[x]['velocidad_maxima'] / resultados_equipos[x]['velocidad_promedio'] - 1) * 100,
                               reverse=True)

    print(f"{'Pos':<4} {'Equipo':<15} {'Eficiencia':<12} {'Vmax':<10}")
    print("-"*60)
    for i, equipo in enumerate(ranking_eficiencia, 1):
        datos = resultados_equipos[equipo]
        eficiencia = (datos['velocidad_maxima'] / datos['velocidad_promedio'] - 1) * 100
        color_code = "\033[92m" if i == 1 else "\033[93m" if i == 2 else "\033[91m" if i == 3 else ""
        reset_code = "\033[0m" if i <= 3 else ""

        print(f"{color_code}{i:<4} {equipo:<15} {eficiencia:<11.1f}% {datos['velocidad_maxima']:<9.1f}{reset_code}")

    # Guardar gráfico
    out_dir = "output/figures"
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{out_dir}/eficiencia_general_{evento['EventName'].replace(' ','_')}_{year}_{sesion_tipo}.png"

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n💾 Gráfico guardado en: {filename}")

    plt.show()

# ==========================================================================
# Verificación de disponibilidad de datos
# ==========================================================================
def verificar_disponibilidad_datos():
    """
    Verifica si los datos de una sesión específica están disponibles.
    Utiliza las funciones existentes elegir_gp() y elegir_sesion()
    """
    try:
        print(f"\n🔍 VERIFICADOR DE DISPONIBILIDAD DE DATOS")
        print("=" * 50)

        # Usar las funciones existentes para seleccionar sesión
        year = int(input("Año de la temporada (ej: 2024, 2025): "))
        evento = elegir_gp(year)
        sesion_tipo = elegir_sesion(evento)

        print(f"\n📡 Verificando: {evento['EventName']} {year} - {sesion_tipo}")
        print("-" * 50)

        # Obtener la sesión usando RoundNumber
        session = fastf1.get_session(year, int(evento["RoundNumber"]), sesion_tipo)

        # Diccionario para almacenar resultados
        disponibilidad = {
            'sesion_encontrada': False,
            'datos_basicos': False,
            'telemetria': False,
            'vueltas_validas': 0,
            'pilotos_presentes': [],
            'error': None,
            'evento': evento['EventName'],
            'year': year,
            'sesion_tipo': sesion_tipo
        }

        # 1. Verificar si la sesión existe
        try:
            session.load(telemetry=False, laps=False, weather=False, messages=False)
            disponibilidad['sesion_encontrada'] = True
            print("✅ Sesión encontrada en el sistema")
        except Exception as e:
            disponibilidad['error'] = f"Sesión no encontrada: {e}"
            print(f"❌ Sesión no encontrada: {e}")
            return disponibilidad

        # 2. Verificar datos básicos (laps)
        try:
            session.load(laps=True, telemetry=False, weather=False)
            laps = session.laps
            if len(laps) > 0:
                disponibilidad['datos_basicos'] = True
                disponibilidad['vueltas_validas'] = len(laps)
                pilotos = laps['Driver'].unique()
                disponibilidad['pilotos_presentes'] = list(pilotos)

                print(f"✅ Datos básicos disponibles")
                print(f"   • Vueltas registradas: {len(laps)}")
                print(f"   • Pilotos presentes: {', '.join(pilotos)}")
            else:
                print("⚠️ Sesión encontrada pero sin vueltas registradas")

        except Exception as e:
            print(f"❌ Error cargando datos básicos: {e}")

        # 3. Verificar telemetría
        try:
            # Intentar cargar telemetría de una vuelta aleatoria
            if len(session.laps) > 0:
                sample_lap = session.laps.iloc[0]
                telemetry = sample_lap.get_telemetry()

                if telemetry is not None and len(telemetry) > 0:
                    disponibilidad['telemetria'] = True
                    print(f"✅ Telemetría disponible")
                    print(f"   • Puntos de datos: {len(telemetry)}")
                else:
                    print("⚠️ Telemetría no disponible aún")
            else:
                print("⚠️ No hay vueltas para verificar telemetría")

        except Exception as e:
            print(f"❌ Error cargando telemetría: {e}")

        # Resumen final
        print("-" * 50)
        if disponibilidad['datos_basicos']:
            print("🎯 ESTADO: Datos básicos LISTOS para análisis")
            if disponibilidad['telemetria']:
                print("       + Telemetría DISPONIBLE - Análisis completo posible")
            else:
                print("       - Telemetría NO disponible - Solo análisis básico")
        else:
            print("💤 ESTADO: Datos NO disponibles aún")

        return disponibilidad

    except Exception as e:
        error_msg = f"Error general: {e}"
        print(f"❌ {error_msg}")
        return {
            'sesion_encontrada': False,
            'datos_basicos': False,
            'telemetria': False,
            'vueltas_validas': 0,
            'pilotos_presentes': [],
            'error': error_msg
        }

def monitor_disponibilidad_automatico():
    """Monitor automático que verifica cada 5 minutos hasta que los datos estén disponibles"""
    import time

    print(f"\n🔍 MONITOR AUTOMÁTICO DE DISPONIBILIDAD")
    print("=" * 50)

    # Usar las funciones existentes para seleccionar sesión
    year = int(input("Año de la temporada (ej: 2024, 2025): "))
    evento = elegir_gp(year)
    sesion_tipo = elegir_sesion(evento)

    print(f"\n🎯 Monitorando: {evento['EventName']} {year} - {sesion_tipo}")
    print("Este script verificará cada 5 minutos hasta que los datos estén disponibles")
    print("Presiona Ctrl+C para detener\n")

    intentos = 0
    while True:
        intentos += 1
        print(f"\n📡 Intento #{intentos} - {time.strftime('%H:%M:%S')}")

        try:
            session = fastf1.get_session(year, int(evento["RoundNumber"]), sesion_tipo)
            session.load(laps=True, telemetry=False)

            if len(session.laps) > 0:
                print(f"🎉 ¡DATOS DISPONIBLES! - {len(session.laps)} vueltas registradas")
                print("Puedes comenzar tu análisis.")
                break
            else:
                print(f"⏳ Datos no disponibles aún. Próxima verificación en 5 minutos...")
                time.sleep(300)  # Esperar 5 minutos

        except Exception as e:
            print(f"❌ Error: {e}")
            print(f"⏳ Reintentando en 5 minutos...")
            time.sleep(300)  # Esperar 5 minutos

# ==========================================================================
def salir():
    """Función mejorada para salir del programa"""
    print("\n" + "🎌" * 25)
    print("   ¡Gracias por usar el analizador F1!")
    print("   Desarrollado para amantes del motorsport 🏎️")
    print("🎌" * 25)

def mostrar_banner():
    """Muestra un banner atractivo para el menú principal"""
    banner = """
    ╔═══════════════════════════════════════════════╗
    ║              🏎️  F1 ANALYTICS PRO             ║
    ║           Análisis Avanzado de F1             ║
    ╚═══════════════════════════════════════════════╝
    """
    print(banner)

# ==========================================================================
def menu_principal():
    """Menú principal del programa"""
    while True:
        mostrar_banner()

        print("📊 **ANÁLISIS DE RITMO Y VELOCIDAD**")
        print("┌─────────────────────────────────────────────────┐")
        print("│  🎯 1. Comparar ritmo entre pilotos             │")
        print("│  🏁 2. Ritmo de un piloto específico            │")
        print("│  ⏱️ 3. Tabla de tiempos de vuelta               │")
        print("│  🚀 4. Eficiencia aerodinámica en recta        │")
        print("│  📈 5. Eficiencia General                       │")
        print("├─────────────────────────────────────────────────┤")
        print("│  🔍 6. Verificar disponibilidad de datos        │")
        print("│  📡 7. Monitor automático de disponibilidad     │")
        print("├─────────────────────────────────────────────────┤")
        print("│  ❌ 8. Salir del programa                       │")
        print("└─────────────────────────────────────────────────┘")

        print("\n" + "═" * 50)
        opcion = input("   🎯 Selecciona una opción (1-8): ").strip()
        print("═" * 50)

        if opcion == '1':
            print("\n🚀 Iniciando comparación de ritmo entre pilotos...")
            accion_comparar_pilotos()
        elif opcion == '2':
            print("\n🏎️  Analizando ritmo individual de piloto...")
            accion_piloto_individual()
        elif opcion == '3':
            print("\n⏱️  Generando tabla de tiempos de vuelta...")
            accion_comparar_tiempos_vuelta()
        elif opcion == '4':
            print("\n📊 Analizando eficiencia aerodinámica en recta...")
            accion_eficiencia_aerodinamica_detallada()
        elif opcion == '5':
            print("\n📈 Analizando eficiencia general...")
            accion_eficiencia_general()
        elif opcion == '6':
            print("\n🔍 Verificando disponibilidad de datos...")
            verificar_disponibilidad_datos()
        elif opcion == '7':
            print("\n📡 Iniciando monitor automático...")
            monitor_disponibilidad_automatico()
        elif opcion == '8':
            print("\n" + "✨" * 25)
            print("   ¡Gracias por usar F1 Analytics Pro!")
            print("   ¡Hasta la próxima carrera! 🏁")
            print("✨" * 25)
            break
        else:
            print("\n❌ Opción no válida. Por favor, elige un número del 1 al 8.")

        input("\n   Presiona Enter para continuar...")

if __name__ == "__main__":
    menu_principal()
