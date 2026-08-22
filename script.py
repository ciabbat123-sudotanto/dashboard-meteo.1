import math
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

LAT = 45.4643
LON = 9.1895

# Download Dati da Open-Meteo
url_ens = f"https://ensemble-api.open-meteo.com/v1/ensemble?latitude={LAT}&longitude={LON}&hourly=temperature_850hPa&models=ncep_gefs05,ecmwf_ifs025_ensemble"
res_ens = requests.get(url_ens).json()

url_temp = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=temperature_2m&models=italia_meteo_arpae_icon_2i,meteofrance_arpege_europe,dwd_icon_d2&forecast_days=1"
res_temp = requests.get(url_temp).json()

url_rain = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=rain&models=italia_meteo_arpae_icon_2i,meteofrance_arpege_europe,dwd_icon_d2&forecast_days=1"
res_rain = requests.get(url_rain).json()

url_turb = f"https://api.open-meteo.com/v1/forecast?latitude={LAT}&longitude={LON}&hourly=rain,cape,wind_speed_1000hPa,wind_speed_850hPa,wind_speed_500hPa,wind_direction_1000hPa,wind_direction_850hPa,wind_direction_500hPa&models=italia_meteo_arpae_icon_2i,meteofrance_arpege_europe,dwd_icon_d2&forecast_days=1"
res_turb = requests.get(url_turb).json()

# Layout Dashboard
fig = plt.figure(figsize=(14, 10), facecolor='#f8f9fa')
gs = gridspec.GridSpec(2, 4, height_ratios=[2, 1], hspace=0.35, wspace=0.25)

fig.suptitle("Previsioni Meteo Ciabatta", fontsize=22, fontweight='bold', y=0.96)

# Grafico Ensemble
ax_main = fig.add_subplot(gs[0, :])
if "hourly" in res_ens:
    time_list = pd.to_datetime(res_ens["hourly"]["time"])
    models = ["ncep_gefs05", "ecmwf_ifs025_ensemble"]
    model_means = []

    for mod in models:
        member_cols = [k for k in res_ens["hourly"].keys() if mod in k]
        if member_cols:
            df_mod = pd.DataFrame({col: res_ens["hourly"][col] for col in member_cols})
            mean_single = df_mod.mean(axis=1)
            model_means.append(mean_single)
            
            above_mean = df_mod.apply(lambda row: row[row > row.mean()].mean(), axis=1)
            below_mean = df_mod.apply(lambda row: row[row < row.mean()].mean(), axis=1)

            ax_main.plot(time_list, mean_single, color='gray', linewidth=1.5, label=f'Media ({mod})')
            ax_main.plot(time_list, above_mean, color='red', linestyle='--', linewidth=1.2, label=f'Sopramedia ({mod})')
            ax_main.plot(time_list, below_mean, color='blue', linestyle='--', linewidth=1.2, label=f'Sottomedia ({mod})')

    if len(model_means) > 0:
        overall_mean = pd.concat(model_means, axis=1).mean(axis=1)
        ax_main.plot(time_list, overall_mean, color='black', linewidth=2.5, label='Media Complessiva')

ax_main.set_title("Temperatura 850 hPa - Modelli Ensemble", fontsize=12, pad=8)
ax_main.set_ylabel("Temperatura (°C)")
ax_main.grid(True, linestyle=':', alpha=0.6)
ax_main.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., fontsize=8.5, framealpha=0.9)

# Metriche
models_3 = ["italia_meteo_arpae_icon_2i", "meteofrance_arpege_europe", "dwd_icon_d2"]

temps = [res_temp["hourly"][f"temperature_2m_{mod}"] for mod in models_3 if f"temperature_2m_{mod}" in res_temp.get("hourly", {})]
mean_temp, max_temp, min_temp = 0, 0, 0
if temps:
    arr_temp = np.array(temps)
    mean_temp = np.nanmean(arr_temp)
    max_temp = np.mean([np.nanmax(m) for m in temps])
    min_temp = np.mean([np.nanmin(m) for m in temps])

rains = [res_rain["hourly"][f"rain_{mod}"] for mod in models_3 if f"rain_{mod}" in res_rain.get("hourly", {})]
rain_pct = 0
if rains:
    arr_rain = np.array(rains)
    rain_hours = np.any(arr_rain > 0.1, axis=0)
    rain_pct = int(round((np.sum(rain_hours) / len(rain_hours)) * 100))

def calc_wind_components(speed_kmh, dir_deg):
    speed_ms = speed_kmh / 3.6
    rad = math.radians(dir_deg)
    return -speed_ms * math.sin(rad), -speed_ms * math.cos(rad)

avg_rain_sum, indice = 0, 0
livello, color_box = "Livello 0", "#28a745"

if "hourly" in res_turb:
    h_data = res_turb["hourly"]
    cape_max_list, shear_max_list, rain_sum_list = [], [], []

    for mod in models_3:
        if f"cape_{mod}" in h_data: cape_max_list.append(np.nanmax(h_data[f"cape_{mod}"]))
        if f"rain_{mod}" in h_data: rain_sum_list.append(np.nansum(h_data[f"rain_{mod}"]))

        w1000_s, w1000_d = h_data.get(f"wind_speed_1000hPa_{mod}"), h_data.get(f"wind_direction_1000hPa_{mod}")
        w850_s, w850_d = h_data.get(f"wind_speed_850hPa_{mod}"), h_data.get(f"wind_direction_850hPa_{mod}")
        w500_s, w500_d = h_data.get(f"wind_speed_500hPa_{mod}"), h_data.get(f"wind_direction_500hPa_{mod}")

        if w1000_s and w850_s and w500_s:
            hourly_shears = []
            for i in range(len(w1000_s)):
                u1000, v1000 = calc_wind_components(w1000_s[i], w1000_d[i])
                u850, v850 = calc_wind_components(w850_s[i], w850_d[i])
                u500, v500 = calc_wind_components(w500_s[i], w500_d[i])
                s1 = math.sqrt((u850 - u1000)**2 + (v850 - v1000)**2)
                s2 = math.sqrt((u500 - u850)**2 + (v500 - v850)**2)
                hourly_shears.append(s1 + s2)
            shear_max_list.append(np.nanmax(hourly_shears))

    avg_cape = np.mean(cape_max_list) if cape_max_list else 0
    avg_rain_sum = np.mean(rain_sum_list) if rain_sum_list else 0
    avg_shear = np.mean(shear_max_list) if shear_max_list else 0

    indice = (avg_cape * avg_shear * avg_rain_sum) / 100000.0

    if 0 <= indice <= 4: livello, color_box = "Livello 0", "#28a745"
    elif 4 < indice <= 9: livello, color_box = "Livello 1", "#ffc107"
    elif 10 <= indice <= 19: livello, color_box = "Livello 2", "#fd7e14"
    else: livello, color_box = "Livello 3", "#dc3545"

# Caselle KPI
ax_kpi1 = fig.add_subplot(gs[1, 0]); ax_kpi1.axis('off')
ax_kpi1.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, color='white', ec='#ccc', lw=1, transform=ax_kpi1.transAxes))
ax_kpi1.text(0.05, 0.85, "Temperatura (°C)", fontsize=10, fontweight='bold', transform=ax_kpi1.transAxes)
ax_kpi1.text(0.3, 0.35, f"{mean_temp:.1f}°C", fontsize=20, fontweight='bold', color='black', ha='center', transform=ax_kpi1.transAxes)
ax_kpi1.text(0.3, 0.18, "Media", fontsize=8, color='gray', ha='center', transform=ax_kpi1.transAxes)
ax_kpi1.text(0.75, 0.52, f"▲ {max_temp:.1f}°C", fontsize=10, fontweight='bold', color='red', ha='center', transform=ax_kpi1.transAxes)
ax_kpi1.text(0.75, 0.22, f"▼ {min_temp:.1f}°C", fontsize=10, fontweight='bold', color='blue', ha='center', transform=ax_kpi1.transAxes)

ax_kpi2 = fig.add_subplot(gs[1, 1]); ax_kpi2.axis('off')
ax_kpi2.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, color='white', ec='#ccc', lw=1, transform=ax_kpi2.transAxes))
ax_kpi2.text(0.05, 0.85, "Pioggia (%)", fontsize=10, fontweight='bold', transform=ax_kpi2.transAxes)
ax_kpi2.text(0.5, 0.38, f"{rain_pct}%", fontsize=28, fontweight='bold', color='#1E88E5', ha='center', transform=ax_kpi2.transAxes)

ax_kpi3 = fig.add_subplot(gs[1, 2]); ax_kpi3.axis('off')
ax_kpi3.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, color='white', ec='#ccc', lw=1, transform=ax_kpi3.transAxes))
ax_kpi3.text(0.05, 0.85, "Massa d'acqua (mm)", fontsize=10, fontweight='bold', transform=ax_kpi3.transAxes)
ax_kpi3.text(0.5, 0.38, f"{avg_rain_sum:.1f} mm", fontsize=24, fontweight='bold', color='#0D47A1', ha='center', transform=ax_kpi3.transAxes)

ax_kpi4 = fig.add_subplot(gs[1, 3]); ax_kpi4.axis('off')
ax_kpi4.add_patch(plt.Rectangle((0, 0), 1, 1, fill=True, color=color_box, ec='#ccc', lw=1, transform=ax_kpi4.transAxes))
ax_kpi4.text(0.5, 0.75, "Indice Turboloso", fontsize=10, fontweight='bold', color='white', ha='center', transform=ax_kpi4.transAxes)
ax_kpi4.text(0.5, 0.45, f"{livello}", fontsize=18, fontweight='bold', color='white', ha='center', transform=ax_kpi4.transAxes)
ax_kpi4.text(0.5, 0.20, f"Valore: {indice:.2f}", fontsize=9, color='white', ha='center', transform=ax_kpi4.transAxes)

fig.text(0.05, 0.02, "Dati da Open-Meteo", fontsize=9, color='gray')

# Salvataggio locale su repository
plt.savefig("dashboard_meteo_ciabatta.png", dpi=150, bbox_inches='tight')
print("✅ Immagine generata!")
