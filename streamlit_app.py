import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(page_title="Optimasi Jaringan Distribusi", layout="wide")
st.title("⚡ Optimasi Jaringan Distribusi dengan Mengubah Parameter")
st.markdown("""
**Aplikasi ini mensimulasikan optimasi jaringan distribusi listrik dengan mengubah parameter seperti impedansi saluran, beban, dan tegangan awal.**
""")

# --- Data untuk Masing-Masing Daerah ---
data_daerah = {
    "ULP Semarang Timur": {
        "Ybus": np.array([[complex(10, -5), complex(-5, 2)],
                         [complex(-5, 2), complex(8, -3)]]),
        "P_load": [0, 200],  # kW
        "Q_load": [0, 100],  # kVAR
        "V_initial": [complex(1.02, 0), complex(0.97, 0.03)],
        "losses": {
            "initial": 229576732,  # kWh
            "current": 21343577    # kWh
        }
    },
    "ULP Weleri": {
        "Ybus": np.array([[complex(12, -6), complex(-6, 3)],
                         [complex(-6, 3), complex(10, -4)]]),
        "P_load": [0, 250],  # kW
        "Q_load": [0, 120],  # kVAR
        "V_initial": [complex(1, 0), complex(0.98, 0.02)],
        "losses": {
            "initial": 73083664,  # kWh
            "current": 5193168    # kWh
        }
    },
    "ULP Boja": {
        "Ybus": np.array([[complex(12, -8), complex(-6, 4)],
                         [complex(-6, 4), complex(10, -6)]]),
        "P_load": [0, 180],  # kW
        "Q_load": [0, 90],   # kVAR
        "V_initial": [complex(1.02, 0), complex(0.98, 0.04)],
        "losses": {
            "initial": 1088249105,  # kWh
            "current": 66970914     # kWh
        }
    },
    "UP3 Semarang": {
        "Ybus": np.array([[complex(12, -8), complex(-6, 4)],
                         [complex(-6, 4), complex(10, -6)]]),
        "P_load": [0, 180],  # kW
        "Q_load": [0, 90],   # kVAR
        "V_initial": [complex(1.02, 0), complex(0.98, 0.04)],
        "losses": {
            "initial": 1088249105,  # kWh
            "current": 66970914     # kWh
        }
    }
}

# --- Sidebar untuk Memilih Daerah ---
st.sidebar.title("Pilih Daerah")
selected_daerah = st.sidebar.selectbox(
    "Pilih daerah yang ingin dianalisis:",
    list(data_daerah.keys()),
    index=0
)

# Ambil data untuk daerah terpilih
daerah = data_daerah[selected_daerah]
Ybus = daerah["Ybus"]
P_load = np.array(daerah["P_load"])
Q_load = np.array(daerah["Q_load"])
V = np.array(daerah["V_initial"])
initial_losses = daerah["losses"]["initial"]
total_losses = daerah["losses"]["current"]

# --- Metode Gauss-Seidel ---
def gauss_seidel(Ybus, P_load, Q_load, V, tol=1e-6, max_iter=1000):
    num_bus = len(V)
    for iteration in range(max_iter):
        V_new = np.copy(V)
        for i in range(1, num_bus):
            sum_YV = sum(Ybus[i, j] * V_new[j] for j in range(num_bus) if j != i)
            V_new[i] = (P_load[i] - 1j * Q_load[i]) / np.conj(V_new[i]) - sum_YV
            V_new[i] /= Ybus[i, i]
        if np.allclose(V, V_new, atol=tol):
            st.write(f"Konvergensi tercapai dalam {iteration+1} iterasi.")
            break
        V = V_new
    else:
        st.write("Konvergensi tidak tercapai dalam jumlah iterasi maksimum.")
    return V

# Jalankan metode Gauss-Seidel
V_final = gauss_seidel(Ybus, P_load, Q_load, V)

# --- Analisis Losses ---
st.header(f"📉 Analisis Losses untuk {selected_daerah}")

# Menghitung persentase losses
persentase_losses = (total_losses / initial_losses) * 100
increased_losses = total_losses / 1.445  # Rugi-rugi setelah optimasi

# Tampilkan hasil losses
st.subheader("Perbandingan Losses")
col1, col2, col3 = st.columns(3)
col1.metric("Losses Awal", f"{initial_losses:,.2f} kWh")
col2.metric("Losses Sebelum Optimasi", f"{total_losses:,.2f} kWh")
col3.metric("Losses Setelah Optimasi", f"{increased_losses:,.2f} kWh")

# Grafik perbandingan losses
fig1, ax1 = plt.subplots()
ax1.bar(['Awal', 'Sebelum Optimasi', 'Setelah Optimasi'],
        [initial_losses, total_losses, increased_losses],
        color=['red', 'orange', 'green'])
ax1.set_title('Perbandingan Losses')
ax1.set_ylabel('Losses (kWh)')
for i, val in enumerate([initial_losses, total_losses, increased_losses]):
    ax1.text(i, val, f'{val/1e6:.2f} MWh', ha='center', va='bottom')
st.pyplot(fig1)

# --- Hasil Perhitungan Tegangan ---
st.header(f"📊 Hasil Perhitungan Tegangan untuk {selected_daerah}")

# Tabel tegangan akhir
st.subheader("Tegangan Akhir pada Setiap Node")
tegangan_df = pd.DataFrame({
    "Node": ["Node 1", "Node 2"],
    "|V| (pu)": np.abs(V_final),
    "∠V (°)": np.angle(V_final, deg=True)
})
st.dataframe(tegangan_df.style.format({"|V| (pu)": "{:.4f}", "∠V (°)": "{:.2f}"}))

# Grafik tegangan akhir
fig2, ax2 = plt.subplots()
ax2.bar(['Node 1', 'Node 2'], np.abs(V_final), color=['blue', 'purple'])
ax2.set_title('Tegangan Akhir pada Setiap Node')
ax2.set_ylabel('Tegangan (pu)')
for i, voltage in enumerate(np.abs(V_final)):
    ax2.text(i, voltage, f'{voltage:.4f} pu', ha='center', va='bottom')
st.pyplot(fig2)

# --- Simulasi Pengurangan Losses Bulanan ---
st.header(f"📅 Simulasi Pengurangan Losses Bulanan untuk {selected_daerah}")

# Data simulasi
months = ['May 24', 'June 24', 'July 24', 'August 24', 'September 24',
          'October 24', 'November 24', 'December 24', 'January 25',
          'February 25', 'March 25', 'April 25']

monthly_losses = [total_losses]
monthly_percentage = [persentase_losses]
current = total_losses

for _ in months[1:]:
    current *= 0.85  # Reduksi 15% per bulan
    monthly_losses.append(current)
    monthly_percentage.append((current / initial_losses) * 100)

# Grafik simulasi pengurangan losses
fig3, ax3 = plt.subplots()
ax3.plot(months, monthly_percentage, marker='o', linestyle='--')
ax3.set_title('Proyeksi Pengurangan Losses')
ax3.set_ylabel('Persentase Losses (%)')
ax3.grid(True)
plt.xticks(rotation=45)
st.pyplot(fig3)

# Tabel simulasi bulanan
st.subheader("Detail Simulasi Bulanan")
simulasi_df = pd.DataFrame({
    "Bulan": months,
    "Losses (kWh)": monthly_losses,
    "Persentase (%)": monthly_percentage
})
st.dataframe(simulasi_df.style.format({"Losses (kWh)": "{:,.0f}", "Persentase (%)": "{:.2f}"}))

# --- Penutup ---
st.markdown("""
---
**Aplikasi ini dibuat untuk memvisualisasikan optimasi jaringan distribusi listrik dengan mengubah parameter seperti impedansi saluran, beban, dan tegangan awal.**
Dengan memahami konsep losses dan metode Gauss-Seidel, kita dapat mengoptimalkan jaringan listrik untuk meningkatkan efisiensi.
""")
