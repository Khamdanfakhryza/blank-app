import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Konfigurasi halaman
st.set_page_config(page_title="Optimasi Jaringan Distribusi", layout="wide")
st.title("⚡ Optimasi Jaringan Distribusi")
st.markdown("""
**Aplikasi ini mensimulasikan optimasi jaringan distribusi listrik menggunakan metode Gauss-Seidel.**
Pilih mode predefined daerah atau input parameter kustom menggunakan sidebar.
""")

# --- Data Predefined untuk Daerah ---
data_daerah = {
    "ULP Semarang Timur": {
        "Ybus": np.array([[complex(10, -5), complex(-5, 2)],
                         [complex(-5, 2), complex(8, -3)]]),
        "P_load": [0, 200],  # kW
        "Q_load": [0, 100],  # kVAR
        "V_initial": [complex(1.02, 0), complex(0.97, 0.03)],
        "losses": {"initial": 229576732, "current": 21343577}
    },
    "ULP Weleri": {
        "Ybus": np.array([[complex(12, -6), complex(-6, 3)],
                         [complex(-6, 3), complex(10, -4)]]),
        "P_load": [0, 250],  # kW
        "Q_load": [0, 120],  # kVAR
        "V_initial": [complex(1, 0), complex(0.98, 0.02)],
        "losses": {"initial": 73083664, "current": 5193168}
    }
}

# --- Sidebar untuk Mode Input ---
st.sidebar.title("Pengaturan Input")
mode = st.sidebar.radio("Pilih Mode Input:", ["Predefined", "Custom"])

if mode == "Predefined":
    # Pilih daerah predefined
    selected_daerah = st.sidebar.selectbox(
        "Pilih daerah:",
        list(data_daerah.keys())
    
    # Ambil data dari daerah terpilih
    daerah = data_daerah[selected_daerah]
    Ybus = daerah["Ybus"]
    P_load = np.array(daerah["P_load"])
    Q_load = np.array(daerah["Q_load"])
    V = np.array(daerah["V_initial"])
    initial_losses = daerah["losses"]["initial"]
    total_losses = daerah["losses"]["current"]

else:
    # Input parameter kustom
    st.sidebar.header("Parameter Jaringan")
    
    # Input Ybus
    st.sidebar.subheader("Matriks Admitansi (Ybus)")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        Y11_real = st.number_input("Y11 (Real)", value=10.0)
        Y11_imag = st.number_input("Y11 (Imaginer)", value=-5.0)
        Y12_real = st.number_input("Y12 (Real)", value=-5.0)
        Y12_imag = st.number_input("Y12 (Imaginer)", value=2.0)
    with col2:
        Y22_real = st.number_input("Y22 (Real)", value=8.0)
        Y22_imag = st.number_input("Y22 (Imaginer)", value=-3.0)
    
    Ybus = np.array([[complex(Y11_real, Y11_imag), complex(Y12_real, Y12_imag)],
                    [complex(Y12_real, Y12_imag), complex(Y22_real, Y22_imag)]])
    
    # Input beban
    st.sidebar.subheader("Beban pada Node 2")
    P_load = [0, st.sidebar.number_input("Daya Nyata (P) - kW", value=200.0)]
    Q_load = [0, st.sidebar.number_input("Daya Reaktif (Q) - kVAR", value=100.0)]
    
    # Input tegangan
    st.sidebar.subheader("Tegangan Awal")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        V1_real = st.number_input("V1 (Real)", value=1.02)
        V1_imag = st.number_input("V1 (Imaginer)", value=0.0)
    with col2:
        V2_real = st.number_input("V2 (Real)", value=0.97)
        V2_imag = st.number_input("V2 (Imaginer)", value=0.03)
    
    V = np.array([complex(V1_real, V1_imag), complex(V2_real, V2_imag)])
    
    # Input losses
    st.sidebar.subheader("Data Losses")
    initial_losses = st.sidebar.number_input("Losses Awal (kWh)", value=229576732)
    total_losses = st.sidebar.number_input("Losses Saat Ini (kWh)", value=21343577)

# --- Metode Gauss-Seidel ---
def gauss_seidel(Ybus, P_load, Q_load, V, tol=1e-6, max_iter=100):
    num_bus = len(V)
    P_load = np.array(P_load)/1000  # Convert kW to MW
    Q_load = np.array(Q_load)/1000  # Convert kVAR to MVAR
    
    history = []
    for iteration in range(max_iter):
        V_new = np.copy(V)
        for i in range(1, num_bus):
            S = complex(P_load[i], Q_load[i])
            sum_YV = sum(Ybus[i,j] * V_new[j] for j in range(num_bus) if j != i)
            V_new[i] = (S.conjugate()/V_new[i].conjugate() - sum_YV) / Ybus[i,i]
        history.append(np.copy(V_new))
        
        if np.max(np.abs(V - V_new)) < tol:
            st.success(f"Konvergensi tercapai dalam {iteration+1} iterasi")
            break
        V = V_new
    else:
        st.error("Konvergensi tidak tercapai dalam iterasi maksimum")
    
    return V, history

# Jalankan Gauss-Seidel
V_final, history = gauss_seidel(Ybus, P_load, Q_load, V)

# --- Visualisasi Hasil ---
st.header("📊 Hasil Simulasi")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Tegangan Node")
    fig, ax = plt.subplots()
    ax.plot([np.abs(v[1]) for v in history], marker='o')
    ax.set_title('Konvergensi Tegangan Node 2')
    ax.set_xlabel('Iterasi')
    ax.set_ylabel('Tegangan (pu)')
    ax.grid(True)
    st.pyplot(fig)

with col2:
    st.subheader("Tegangan Akhir")
    volt_df = pd.DataFrame({
        "Node": ["Node 1", "Node 2"],
        "|V| (pu)": np.abs(V_final),
        "∠V (°)": np.angle(V_final, deg=True)
    })
    st.dataframe(volt_df.style.format({"|V| (pu)": "{:.4f}", "∠V (°)": "{:.2f}"}))

# --- Analisis Losses ---
st.header("📉 Analisis Losses")
increased_losses = total_losses / 1.445  # Contoh optimasi

cols = st.columns(3)
cols[0].metric("Losses Awal", f"{initial_losses/1e6:.2f} MWh")
cols[1].metric("Losses Saat Ini", f"{total_losses/1e6:.2f} MWh")
cols[2].metric("Estimasi Post-Optimasi", f"{increased_losses/1e6:.2f} MWh", "-14.5%")

# Grafik perbandingan
fig, ax = plt.subplots()
labels = ['Awal', 'Saat Ini', 'Post-Optimasi']
values = [initial_losses, total_losses, increased_losses]
ax.bar(labels, values, color=['#ff5555', '#ffb86c', '#50fa7b'])
ax.set_title('Perbandingan Losses')
ax.set_ylabel('Energi (kWh)')
st.pyplot(fig)

# --- Simulasi Bulanan ---
st.header("📅 Proyeksi Pengurangan Losses")
months = ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 
          'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']

current_loss = total_losses
simulasi = []
for _ in months:
    simulasi.append(current_loss)
    current_loss *= 0.85  # Reduksi 15% per bulan

fig, ax = plt.subplots()
ax.plot(months, simulasi, marker='o', linestyle='--')
ax.set_title('Proyeksi Pengurangan Losses')
ax.set_ylabel('Losses (kWh)')
ax.grid(True)
st.pyplot(fig)

# --- Footer ---
st.markdown("""
---
**Aplikasi ini dikembangkan untuk demonstrasi optimasi jaringan distribusi listrik.**
*Teknologi digunakan: Streamlit, NumPy, Matplotlib*
""")
