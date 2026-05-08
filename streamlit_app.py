import streamlit as st
import math
import statistics
import pandas as pd
import numpy as np
import altair as alt
import os
import json
from collections import Counter

# ==========================================
# LÓGICA DEL BACKEND (INTEGRADA E INDEPENDIENTE)
# ==========================================

# --- INDICADORES ---
def calcular_moda(datos):
    conteos = Counter(datos)
    max_frecuencia = max(conteos.values())
    modas = [k for k, v in conteos.items() if v == max_frecuencia]
    return modas

def calcular_percentil(datos_ordenados, p):
    n = len(datos_ordenados)
    if n == 0: return 0.0
    if n == 1: return datos_ordenados[0]
    idx = (p / 100) * (n - 1)
    idx_inf = int(math.floor(idx))
    idx_sup = int(math.ceil(idx))
    peso = idx - idx_inf
    if idx_inf == idx_sup:
        return datos_ordenados[idx_inf]
    else:
        return datos_ordenados[idx_inf] * (1 - peso) + datos_ordenados[idx_sup] * peso

# --- CONTEO ---
def permutacion_sin_repeticion(n): return math.factorial(n)
def permutacion_con_repeticion(n, r): return n ** r
def combinacion_sin_repeticion(n, r): return math.comb(n, r) if r <= n else 0
def combinacion_con_repeticion(n, r): return math.comb(n + r - 1, r)
def variacion_sin_repeticion(n, r): return math.factorial(n) // math.factorial(n - r) if r <= n else 0
def variacion_con_repeticion(n, r): return n ** r

# --- DISTRIBUCIONES ---
def funcion_error_aprox(x):
    signo = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * math.exp(-x * x)
    return signo * y

def cdf_normal_estandar(z): return 0.5 * (1.0 + funcion_error_aprox(z / math.sqrt(2)))
def pdf_normal(x, mu, sigma): return (1.0 / (sigma * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)
def probabilidad_binomial(k, n, p): return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
def probabilidad_poisson(k, lam): return (math.exp(-lam) * (lam ** k)) / math.factorial(k)

# --- INTERVALOS ---
def get_z_value(confianza):
    alfa = 1.0 - confianza
    p = 1 - alfa / 2
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - ((c2*t + c1)*t + c0) / (((d3*t + d2)*t + d1)*t + 1.0)
    return z

def get_t_value(confianza, df):
    z = get_z_value(confianza)
    if df >= 30: return z
    return z + (z**3 + z) / (4.0 * df)

# --- SISTEMA DE PERSISTENCIA ---
def registrar_calculo(modulo, inputs_dict):
    try:
        log_entry = {
            "fecha": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "modulo": modulo,
            "inputs": inputs_dict
        }
        with open("historial_calculos.json", "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
        return True
    except:
        return False

# ==========================
# CONFIGURACIÓN GENERAL & UI
# ==========================
st.set_page_config(
    page_title="Statistical Solver Pro | Edición Premium", 
    page_icon="🧬", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Inyección de CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    .stApp {
        background: radial-gradient(circle at 50% 50%, #111827 0%, #030712 100%);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    .stDeployButton, #MainMenu, footer { display: none !important; }
    [data-testid="stToolbar"] { display: flex !important; background: transparent !important; }
    header[data-testid="stHeader"] { background: transparent !important; visibility: visible !important; }

    [data-testid="stExpandSidebarButton"], 
    button[data-testid="stBaseButton-headerNoPadding"] {
        visibility: visible !important;
        display: flex !important;
        background: rgba(255, 255, 255, 0.1) !important;
        border-radius: 50% !important;
        z-index: 1000002 !important;
    }

    [data-testid="stExpandSidebarButton"] svg, 
    button[data-testid="stBaseButton-headerNoPadding"] svg {
        fill: white !important;
        color: white !important;
    }

    .glass-card {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 24px;
        padding: 30px;
        margin-top: 10px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
    }
    
    .hero-title {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -0.05em;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 15px !important;
    }
</style>
""", unsafe_allow_html=True)

def render_section_header(title, subtitle, icon="📊"):
    st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 20px; margin-bottom: 25px;">
            <div style="font-size: 2.5rem; background: rgba(56, 189, 248, 0.1); padding: 12px; border-radius: 20px;">{icon}</div>
            <div>
                <h1 style="margin: 0; font-size: 2rem; font-weight: 800;">{title}</h1>
                <p style="margin: 0; color: #64748b;">{subtitle}</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# ==========================
# GESTIÓN DE NAVEGACIÓN
# ==========================
OPCIONES_MENU = [
    "🏠 Panel de Control",
    "1. Indicadores Estadísticos",
    "2. Reglas de Conteo",
    "3. Teorema de Bayes",
    "4. Variable Aleatoria",
    "5. Distribuciones Probabilísticas",
    "6. Intervalos de Confianza",
    "📜 Historial de Cálculos"
]

if "nav_index" not in st.session_state:
    st.session_state["nav_index"] = 0

# ==========================
# MENÚ LATERAL (SIDEBAR)
# ==========================
with st.sidebar:
    st.markdown('<div style="text-align: center; padding-bottom: 20px;"><h3>🧠 INTELIGENCIA ESTADISTICA</h3></div>', unsafe_allow_html=True)
    st.markdown("---")
    opcion = st.radio("SISTEMA DE ANÁLISIS", OPCIONES_MENU, index=st.session_state["nav_index"])

# =======================================================
# 🏠 PANEL DE CONTROL
# =======================================================
if "Panel" in opcion:
    st.markdown('<h1 class="hero-title">Inteligencia Estadística</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #64748b; margin-bottom: 30px;">Plataforma de alta fidelidad para el análisis y modelado de datos académicos.</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-card" style="border-top: 4px solid #38bdf8;"><h3>Integridad</h3><p>Algoritmos sincronizados con las librerías originales del proyecto.</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="glass-card" style="border-top: 4px solid #c084fc;"><h3>Reactividad</h3><p>Procesamiento instantáneo en tiempo real sin esperas.</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card"><h3>Estado del Laboratorio</h3><p>Seleccione una herramienta para comenzar.</p></div>', unsafe_allow_html=True)

# =======================================================
# 1. INDICADORES
# =======================================================
elif "1." in opcion:
    render_section_header("Indicadores Estadísticos", "Medidas de tendencia, dispersión, forma y posición.", "🧮")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    sub = st.selectbox("Seleccione Categoría:", ["Tendencia Central", "Dispersión", "Forma (Pearson)", "Posición"], key="ind_sub")
    
    if sub in ["Tendencia Central", "Dispersión", "Posición"]:
        met = st.radio("Método:", ["Datos Brutos", "Datos Agrupados"], horizontal=True, key="ind_met") if sub == "Posición" else "Datos Brutos"
        
        if met == "Datos Brutos":
            d_str = st.text_input("Ingrese Datos (Xi):", "12, 14, 10, 15, 14, 18", key="ind_xi")
            if sub == "Dispersión": t_d = st.radio("Contexto:", ["Muestra", "Población"], horizontal=True, key="ind_td")
            
            try:
                datos = sorted([float(x) for x in d_str.replace(',',' ').split() if x.strip()])
                if len(datos) >= 2:
                    if sub == "Tendencia Central":
                        m, med = statistics.mean(datos), statistics.median(datos)
                        modas = calcular_moda(datos)
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Media", f"{m:.4f}")
                        c2.metric("Mediana", f"{med:.4f}")
                        c3.metric("Moda(s)", ", ".join(f"{x:g}" for x in modas))
                        if st.button("📝 Registrar"): registrar_calculo("Tendencia Central", {"ind_sub":"Tendencia Central","ind_xi":d_str})
                    elif sub == "Dispersión":
                        m, dev = statistics.mean(datos), (statistics.pstdev(datos) if t_d=="Población" else statistics.stdev(datos))
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Varianza", f"{dev**2:.4f}")
                        c2.metric("Desv. Est.", f"{dev:.4f}")
                        c3.metric("CV", f"{(dev/m*100):.2f}%")
                        if st.button("📝 Registrar"): registrar_calculo("Dispersión", {"ind_sub":"Dispersión","ind_xi":d_str,"ind_td":t_d})
                    elif sub == "Posición":
                        st.markdown("---")
                        pk = st.number_input("Percentil (k):", 1, 99, 90, key="ind_pk")
                        st.metric(f"P{pk}", f"{calcular_percentil(datos, pk):.4f}")
                        if st.button("📝 Registrar"): registrar_calculo("Posición Brutos", {"ind_sub":"Posición","ind_met":"Datos Brutos","ind_xi":d_str,"ind_pk":pk})
            except: pass
        else:
            tipo_p = st.radio("Tipo:", ["Percentil", "Cuartil"], horizontal=True, key="tk_tipo")
            tk_k = st.number_input("k:", 1, 99, 60, key="tk_k")
            c1, c2, c3 = st.columns(3)
            n, a, l = c1.number_input("N:", 1.0, value=100.0, key="tk_n"), c2.number_input("ai:", 0.1, value=10.0, key="tk_a"), c3.number_input("Li:", value=0.0, key="tk_l")
            f, na = c1.number_input("ni:", 0.1, value=10.0, key="tk_f"), c2.number_input("Ni-1:", value=0.0, key="tk_na")
            res = l + (((tk_k*n)/(100 if tipo_p=="Percentil" else 4))-na)/f*a
            st.metric("Resultado", f"{res:.4f}")
            if st.button("📝 Registrar"): registrar_calculo("Posición Agrupados", {"ind_sub":"Posición","ind_met":"Datos Agrupados","tk_tipo":tipo_p,"tk_k":tk_k,"tk_n":n,"tk_a":a,"tk_l":l,"tk_f":f,"tk_na":na})

    elif sub == "Forma (Pearson)":
        c1, c2, c3 = st.columns(3)
        m, mo, s = c1.number_input("Media:", value=15.0, key="f_m"), c2.number_input("Moda:", value=14.0, key="f_mo"), c3.number_input("Desv:", min_value=0.1, value=2.0, key="f_s")
        asimetria = (3*(m-mo))/s
        st.metric("Asimetría", f"{asimetria:.4f}")
        if st.button("📝 Registrar"): registrar_calculo("Forma", {"ind_sub":"Forma (Pearson)","f_m":m,"f_mo":mo,"f_s":s})
    st.markdown('</div>', unsafe_allow_html=True)

# =======================================================
# 2. REGLAS DE CONTEO
# =======================================================
elif "2." in opcion:
    render_section_header("Reglas de Conteo", "Cálculo de cardinalidad.", "🔢")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    mod = st.selectbox("Modelo:", ["Permutación P(n)", "PR(n,r)", "Combinación C(n,r)", "CR(n,r)", "Variación V(n,r)", "VR(n,r)"], key="c_mod")
    n, r = st.number_input("n:", 1, value=10, key="c_n"), st.number_input("r:", 0, value=3, key="c_r")
    if "P(n)" in mod: res = permutacion_sin_repeticion(n)
    elif "PR" in mod or "VR" in mod: res = permutacion_con_repeticion(n, r)
    elif "CR" in mod: res = combinacion_con_repeticion(n, r)
    elif "C" in mod: res = combinacion_sin_repeticion(n, r)
    else: res = variacion_sin_repeticion(n, r)
    st.metric("Total", f"{res:,}")
    if st.button("📝 Registrar"): registrar_calculo("Conteo", {"c_mod":mod,"c_n":n,"c_r":r})
    st.markdown('</div>', unsafe_allow_html=True)

# =======================================================
# 3. TEOREMA DE BAYES
# =======================================================
elif "3." in opcion:
    render_section_header("Teorema de Bayes", "Probabilidades a posteriori.", "🧠")
    nh = st.number_input("Hipótesis:", 2, 5, 2, key="b_nh")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    p1, p2 = [], []
    for i in range(nh):
        c1, c2 = st.columns(2)
        p1.append(c1.number_input(f"P(A{i+1})", 0.0, 1.0, 1.0/nh, key=f"b_p1_{i}"))
        p2.append(c2.number_input(f"P(B|A{i+1})", 0.0, 1.0, 0.5, key=f"b_p2_{i}"))
    if abs(sum(p1)-1.0) < 1e-4:
        pb = sum(a*b for a,b in zip(p1,p2))
        if pb > 0:
            st.metric("P(B)", f"{pb:.4f}")
            if st.button("📝 Registrar"): registrar_calculo("Bayes", {"b_nh":nh, **{f"b_p1_{i}":v for i,v in enumerate(p1)}, **{f"b_p2_{i}":v for i,v in enumerate(p2)}})
    st.markdown('</div>', unsafe_allow_html=True)

# =======================================================
# 4. VARIABLE ALEATORIA
# =======================================================
elif "4." in opcion:
    render_section_header("Variable Aleatoria", "Momentos discretos.", "🎲")
    nv = st.number_input("Valores:", 1, 5, 3, key="va_nv")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    vx, vp = [], []
    for i in range(nv):
        c1, c2 = st.columns(2)
        vx.append(c1.number_input(f"x_{i+1}", value=float(i), key=f"va_x_{i}"))
        vp.append(c2.number_input(f"P(x_{i+1})", 0.0, 1.0, 1.0/nv, key=f"va_p_{i}"))
    if abs(sum(vp)-1.0) < 1e-4:
        ex = sum(a*b for a,b in zip(vx,vp))
        st.metric("E[X]", f"{ex:.4f}")
        if st.button("📝 Registrar"): registrar_calculo("Var. Aleatoria", {"va_nv":nv, **{f"va_x_{i}":v for i,v in enumerate(vx)}, **{f"va_p_{i}":v for i,v in enumerate(vp)}})
    st.markdown('</div>', unsafe_allow_html=True)

# =======================================================
# 5. DISTRIBUCIONES
# =======================================================
elif "5." in opcion:
    render_section_header("Distribuciones", "Densidad y masa.", "📈")
    dt = st.selectbox("Modelo:", ["Normal", "Poisson", "Binomial", "Exponencial"], key="d_mod")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if dt == "Normal":
        m, s, x = st.number_input("μ:", value=0.0, key="d_norm_m"), st.number_input("σ:", 0.1, value=1.0, key="d_norm_s"), st.number_input("x:", value=0.0, key="d_norm_x")
        res = cdf_normal_estandar((x-m)/s)
        inputs = {"d_mod":dt, "d_norm_m":m, "d_norm_s":s, "d_norm_x":x}
    elif dt == "Binomial":
        n, p, k = st.number_input("n:", 1, key="d_bin_n"), st.number_input("p:", 0.0, 1.0, 0.5, key="d_bin_p"), st.number_input("k:", 0, key="d_bin_k")
        res = probabilidad_binomial(int(k), int(n), p)
        inputs = {"d_mod":dt, "d_bin_n":n, "d_bin_p":p, "d_bin_k":k}
    else: res = 0; inputs = {}
    st.metric("P", f"{res:.4f}")
    if st.button("📝 Registrar"): registrar_calculo("Distribución", inputs)
    st.markdown('</div>', unsafe_allow_html=True)

# =======================================================
# 6. INTERVALOS
# =======================================================
elif "6." in opcion:
    render_section_header("Intervalos", "Límites de confianza.", "📏")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    n, m, s = st.number_input("n:", 2, key="i_n"), st.number_input("x̄:", value=100.0, key="i_m"), st.number_input("s:", 0.1, value=15.0, key="i_s")
    c = st.slider("Confianza:", 0.8, 0.99, 0.95, key="i_c")
    err = get_z_value(c) * (s / math.sqrt(n))
    st.metric("Intervalo", f"[{m-err:.4f}, {m+err:.4f}]")
    if st.button("📝 Registrar"): registrar_calculo("Intervalo", {"i_n":n,"i_m":m,"i_s":s,"i_c":c})
    st.markdown('</div>', unsafe_allow_html=True)

# =======================================================
# 📜 HISTORIAL
# =======================================================
elif "📜" in opcion:
    render_section_header("Historial", "Replicación de cálculos.", "📜")
    path = "historial_calculos.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            registros = [json.loads(line) for line in f]
        for i, reg in enumerate(reversed(registros)):
            with st.expander(f"🕒 {reg['fecha']} - {reg['modulo']}"):
                if st.button(f"🔄 Replicar #{i}", key=f"rep_{i}"):
                    for k, v in reg['inputs'].items(): st.session_state[k] = v
                    # Mapeo de navegación a índice
                    NAV_MAP = {"Tendencia Central":1,"Dispersión":1,"Posición Brutos":1,"Posición Agrupados":1,"Forma":1,"Conteo":2,"Bayes":3,"Var. Aleatoria":4,"Distribución":5,"Intervalo":6}
                    st.session_state["nav_index"] = NAV_MAP.get(reg['modulo'], 0)
                    st.rerun()
        if st.button("🗑️ Borrar"): os.remove(path); st.rerun()
    else: st.info("Vacío.")
