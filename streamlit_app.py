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
def permutacion_lineal_sin_rep(n): return math.factorial(n)
def permutacion_con_rep_identicos(n, frecuencias):
    denominador = 1
    for f in frecuencias: denominador *= math.factorial(f)
    return math.factorial(n) // denominador
def variacion_sin_rep(n, r): return math.factorial(n) // math.factorial(n - r) if r <= n else 0
def variacion_con_rep(n, r): return n ** r
def combinacion_sin_rep(n, r): return math.comb(n, r) if r <= n else 0
def combinacion_con_rep(n, r): return math.comb(n + r - 1, r)

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
    page_title="Sistema de Resolución Estadístico (Académica)", 
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
    st.markdown('<div style="text-align: center; padding-bottom: 20px;"><h3>🧠 RESOLUCIÓN ESTADÍSTICA</h3></div>', unsafe_allow_html=True)
    st.markdown("---")
    opcion = st.radio("SISTEMA DE ANÁLISIS", OPCIONES_MENU, index=st.session_state["nav_index"])

# =======================================================
# 🏠 PANEL DE CONTROL
# =======================================================
if "Panel" in opcion:
    st.markdown('<h1 class="hero-title">Sistema de Resolución Estadística</h1>', unsafe_allow_html=True)
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
    render_section_header("Reglas de Conteo", "Cálculo de cardinalidad y combinatoria.", "🔢")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    mod = st.selectbox("Modelo de Conteo:", [
        "Permutación (Sin repetición)", 
        "Permutación con repetición", 
        "Combinación (Sin repetición)", 
        "Combinación con repetición", 
        "Variación (Sin repetición)", 
        "Variación con repetición"
    ], key="c_mod")

    # Mostrar fórmula según selección
    if "Permutación (Sin repetición)" in mod:
        st.latex(r"P(n, r) = \frac{n!}{(n-r)!}")
    elif "Permutación con repetición" in mod:
        st.latex(r"P'_n(r) = n^r")
    elif "Combinación (Sin repetición)" in mod:
        st.latex(r"C_n^r = \binom{n}{r} = \frac{n!}{r!(n-r)!}")
    elif "Combinación con repetición" in mod:
        st.latex(r"CR_n^r = \binom{n+r-1}{r}")
    elif "Variación (Sin repetición)" in mod:
        st.latex(r"V_n^r = \frac{n!}{(n-r)!}")
    elif "Variación con repetición" in mod:
        st.latex(r"VR_n^r = n^r")

    col1, col2 = st.columns(2)
    with col1:
        n = st.number_input("n (Población total):", 1, value=10, key="c_n")
    
    with col2:
        # Para casos "Sin repetición", r no puede ser mayor que n
        max_r = n if ("(Sin repetición)" in mod) else None
        r = st.number_input("r (Muestra / Selección):", 0, max_value=max_r, value=min(3, n), key="c_r")
    
    res = 0
    if "Permutación (Sin repetición)" in mod: res = variacion_sin_rep(n, r)
    elif "Permutación con repetición" in mod: res = variacion_con_rep(n, r)
    elif "Combinación con repetición" in mod: res = combinacion_con_rep(n, r)
    elif "Combinación (Sin repetición)" in mod: res = combinacion_sin_rep(n, r)
    elif "Variación con repetición" in mod: res = variacion_con_rep(n, r)
    else: res = variacion_sin_rep(n, r)

    st.markdown("---")
    st.metric("Resultado", f"{res:,}")
    
    if st.button("📝 Registrar"): 
        registrar_calculo("Conteo", {"c_mod":mod,"c_n":n,"c_r":r})
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
    render_section_header("Variable Aleatoria", "Análisis detallado de momentos discretos.", "🎲")
    
    col_setup1, col_setup2 = st.columns([1, 2])
    with col_setup1:
        nv = st.number_input("Número de valores (n):", 1, 20, 3, key="va_nv")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    vx, vp = [], []
    st.markdown("##### Ingrese la distribución de probabilidad:")
    
    # Grid para entrada de datos
    for i in range(nv):
        c1, c2 = st.columns(2)
        vx.append(c1.number_input(f"Valor x_{i+1}", value=float(i), key=f"va_x_{i}"))
        
        # Sugerir probabilidad restante para el último valor
        default_p = 0.0
        if i == nv - 1:
            suma_previa = sum(vp)
            default_p = max(0.0, 1.0 - suma_previa)
        else:
            default_p = 1.0 / nv
            
        vp.append(c2.number_input(f"Probabilidad P(x_{i+1})", 0.0, 1.0, default_p, format="%.4f", key=f"va_p_{i}"))

    suma_p = sum(vp)
    if abs(suma_p - 1.0) > 1e-5:
        st.warning(f"⚠ Las probabilidades suman {suma_p:.4f}. Deben sumar 1.0 para resultados exactos.")

    # Cálculos por Valor (x)
    ex = sum(x * p for x, p in zip(vx, vp))
    ex2 = sum((x**2) * p for x, p in zip(vx, vp))
    var_x = max(0.0, ex2 - ex**2)
    sigma_x = math.sqrt(var_x)
    
    # Cálculos por Índice (i) - i empieza en 1
    indices = list(range(1, nv + 1))
    ei = sum(i * p for i, p in zip(indices, vp))
    ei2 = sum((i**2) * p for i, p in zip(indices, vp))
    var_i = max(0.0, ei2 - ei**2)
    sigma_i = math.sqrt(var_i)
    
    # Crear DataFrame para la tabla
    df_data = {
        "i": indices,
        "x": vx,
        "P(x)": vp,
        "x·P(x)": [x * p for x, p in zip(vx, vp)],
        "i·P(x)": [i * p for i, p in zip(indices, vp)],
        "x²·P(x)": [(x**2) * p for x, p in zip(vx, vp)],
        "i²·P(x)": [(i**2) * p for i, p in zip(indices, vp)]
    }
    df = pd.DataFrame(df_data)
    
    st.markdown("### Tabla de Desglose")
    st.dataframe(df.style.format({
        "P(x)": "{:.4f}",
        "x·P(x)": "{:.4f}",
        "i·P(x)": "{:.4f}",
        "x²·P(x)": "{:.4f}",
        "i²·P(x)": "{:.4f}"
    }), use_container_width=True)
    
    # Resultados Estilizados
    st.markdown("---")
    col_res1, col_res2 = st.columns(2)
    
    with col_res1:
        st.markdown("#### 1. Análisis por Valores (x)")
        st.metric("Esperanza E(X)", f"{ex:.6f}")
        st.metric("Varianza Var(X)", f"{var_x:.6f}")
        st.metric("Desv. Estándar σ", f"{sigma_x:.6f}")
        st.caption(f"E(X²) = {ex2:.6f}")

    with col_res2:
        st.markdown("#### 2. Análisis por Índice (i)")
        st.metric("Esperanza E(i)", f"{ei:.6f}")
        st.metric("Varianza Var(i)", f"{var_i:.6f}")
        st.metric("Desv. Estándar σ(i)", f"{sigma_i:.6f}")
        st.caption(f"E(i²) = {ei2:.6f}")

    if st.button("📝 Registrar en Historial"):
        registrar_calculo("Var. Aleatoria", {
            "va_nv": nv, 
            **{f"va_x_{i}": v for i, v in enumerate(vx)}, 
            **{f"va_p_{i}": v for i, v in enumerate(vp)}
        })
        st.success("Cálculo registrado correctamente.")

    st.markdown('</div>', unsafe_allow_html=True)

# =======================================================
# 5. DISTRIBUCIONES
# =======================================================
elif "5." in opcion:
    render_section_header("Distribuciones", "Densidad, masa e intervalos.", "📈")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    dt = st.selectbox("Modelo de Distribución:", ["Normal", "Binomial", "Poisson", "Exponencial"], key="d_mod")
    tipo_calc = st.selectbox("Tipo de Probabilidad:", ["Puntual / Densidad", "Acumulada Inferior (P ≤ x)", "Acumulada Superior (P > x)", "Intervalo (P [a, b])"], key="d_tipo")
    
    res = 0.0
    res_label = "Resultado"
    inputs = {"d_mod": dt, "d_tipo": tipo_calc}
    
    # --- COLUMNAS DE PARÁMETROS ---
    c1, c2 = st.columns(2)
    
    if dt == "Normal":
        mu = c1.number_input("Media (μ):", value=0.0, key="d_norm_m")
        sigma = c2.number_input("Desv. Estándar (σ):", 0.001, value=1.0, key="d_norm_s")
        if tipo_calc == "Intervalo (P [a, b])":
            a = c1.number_input("Límite inferior (a):", value=-1.0, key="d_norm_a")
            b = c2.number_input("Límite superior (b):", value=1.0, key="d_norm_b")
            res = cdf_normal_estandar((b-mu)/sigma) - cdf_normal_estandar((a-mu)/sigma)
            res_label = f"P({a} ≤ X ≤ {b})"
        else:
            x = st.number_input("Valor (x):", value=0.0, key="d_norm_x")
            if "Puntual" in tipo_calc: 
                res = pdf_normal(x, mu, sigma)
                res_label = f"f({x}) [Densidad]"
            elif "Inferior" in tipo_calc: 
                res = cdf_normal_estandar((x-mu)/sigma)
                res_label = f"P(X ≤ {x})"
            else: 
                res = 1 - cdf_normal_estandar((x-mu)/sigma)
                res_label = f"P(X > {x})"
        inputs.update({"mu": mu, "sigma": sigma})

    elif dt == "Binomial":
        n = c1.number_input("Ensayos (n):", 1, value=10, key="d_bin_n")
        p = c2.number_input("Prob. éxito (p):", 0.0, 1.0, 0.5, key="d_bin_p")
        if tipo_calc == "Intervalo (P [a, b])":
            a = c1.number_input("Mínimo éxitos (a):", 0, n, 0, key="d_bin_a")
            b = c2.number_input("Máximo éxitos (b):", 0, n, n, key="d_bin_b")
            res = sum(probabilidad_binomial(i, n, p) for i in range(int(a), int(b) + 1))
            res_label = f"P({a} ≤ X ≤ {b})"
        else:
            k = st.number_input("Éxitos (k):", 0, n, 0, key="d_bin_k")
            if "Puntual" in tipo_calc: 
                res = probabilidad_binomial(int(k), n, p)
                res_label = f"P(X = {k})"
            elif "Inferior" in tipo_calc: 
                res = sum(probabilidad_binomial(i, n, p) for i in range(int(k) + 1))
                res_label = f"P(X ≤ {k})"
            else: 
                res = 1 - sum(probabilidad_binomial(i, n, p) for i in range(int(k) + 1))
                res_label = f"P(X > {k})"
        inputs.update({"n": n, "p": p})

    elif dt == "Poisson":
        lam = c1.number_input("Tasa promedio (λ):", 0.001, value=5.0, key="d_poi_l")
        if tipo_calc == "Intervalo (P [a, b])":
            a = c1.number_input("Mínimo (a):", 0, key="d_poi_a")
            b = c2.number_input("Máximo (b):", 0, key="d_poi_b")
            res = sum(probabilidad_poisson(i, lam) for i in range(int(a), int(b) + 1))
            res_label = f"P({a} ≤ X ≤ {b})"
        else:
            k = st.number_input("Ocurrencias (k):", 0, key="d_poi_k")
            if "Puntual" in tipo_calc: 
                res = probabilidad_poisson(int(k), lam)
                res_label = f"P(X = {k})"
            elif "Inferior" in tipo_calc: 
                res = sum(probabilidad_poisson(i, lam) for i in range(int(k) + 1))
                res_label = f"P(X ≤ {k})"
            else: 
                res = 1 - sum(probabilidad_poisson(i, lam) for i in range(int(k) + 1))
                res_label = f"P(X > {k})"
        inputs.update({"lam": lam})

    elif dt == "Exponencial":
        lam = c1.number_input("Tasa (λ):", 0.001, value=1.0, key="d_exp_l")
        if tipo_calc == "Intervalo (P [a, b])":
            a = c1.number_input("Inicio (a):", 0.0, key="d_exp_a")
            b = c2.number_input("Fin (b):", 0.0, key="d_exp_b")
            res = math.exp(-lam * a) - math.exp(-lam * b)
            res_label = f"P({a} ≤ X ≤ {b})"
        else:
            x = st.number_input("Valor (x):", 0.0, key="d_exp_x")
            if "Puntual" in tipo_calc: 
                res = lam * math.exp(-lam * x)
                res_label = f"f({x}) [Densidad]"
            elif "Inferior" in tipo_calc: 
                res = 1 - math.exp(-lam * x)
                res_label = f"P(X ≤ {x})"
            else: 
                res = math.exp(-lam * x)
                res_label = f"P(X > {x})"
        inputs.update({"lam": lam})

    st.markdown("---")
    
    # --- RESULTADO ESTILIZADO ---
    st.markdown(f"""
        <div style="background: linear-gradient(90deg, rgba(56, 189, 248, 0.2), rgba(56, 189, 248, 0.05)); border: 1px solid rgba(56, 189, 248, 0.3); padding: 25px; border-radius: 20px; text-align: center; margin-bottom: 20px;">
            <h4 style="margin: 0; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.1em; font-size: 0.9rem;">{res_label}</h4>
            <p style="font-size: 2.5rem; font-weight: 800; margin: 10px 0; font-family: 'JetBrains Mono', monospace; color: #f8fafc;">{res:.10f}</p>
        </div>
    """, unsafe_allow_html=True)

    if dt == "Normal":
        c1, c2, c3 = st.columns(3)
        c1.metric("Valor Z", f"{( (x-mu)/sigma if 'Intervalo' not in tipo_calc else 'N/A' )}")
        c2.metric("Varianza", f"{sigma**2:.4f}")
        c3.metric("Desv. Est.", f"{sigma:.4f}")
    elif dt == "Exponencial":
        c1, c2 = st.columns(2)
        c1.metric("Media (1/λ)", f"{1/lam:.4f}")
        c2.metric("Mediana", f"{math.log(2)/lam:.4f}")
    elif dt == "Binomial":
        c1, c2 = st.columns(2)
        c1.metric("Media (np)", f"{n*p:.4f}")
        c2.metric("Varianza (npq)", f"{n*p*(1-p):.4f}")
    elif dt == "Poisson":
        c1, c2 = st.columns(2)
        c1.metric("Media (λ)", f"{lam:.4f}")
        c2.metric("Varianza (λ)", f"{lam:.4f}")
    
    if st.button("📝 Registrar"): registrar_calculo("Distribución", inputs)
    st.markdown('</div>', unsafe_allow_html=True)

# =======================================================
# 6. INTERVALOS
# =======================================================
elif "6." in opcion:
    render_section_header("Intervalos", "Límites de confianza.", "📏")
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        n = st.number_input("Tamaño de muestra (n):", 2, value=30, key="i_n")
        m = st.number_input("Media muestral (x̄):", value=100.0, key="i_m")
        c = st.slider("Nivel de Confianza:", 0.80, 0.99, 0.95, 0.01, key="i_c")
        
    with col_b:
        sigma_conocida = st.radio("¿Conoce σ (poblacional)?", ["Sí, es conocida (σ)", "No, usar muestral (s)"], index=1, key="i_sigma_know")
        label_s = "Desviación Estándar (σ):" if "Sí" in sigma_conocida else "Desviación Estándar (s):"
        s = st.number_input(label_s, 0.01, value=15.0, key="i_s")

    # --- APLICACIÓN DE CRITERIOS ---
    es_sigma = "Sí" in sigma_conocida
    if es_sigma:
        v_critico = get_z_value(c)
        dist_used = "Normal (Z)"
        razon = "Se utiliza la distribución Z porque la desviación estándar poblacional (σ) es conocida."
        color_box = "info"
    else:
        if n >= 30:
            v_critico = get_z_value(c)
            dist_used = "Normal (Z) [Aprox]"
            razon = f"Se utiliza la aproximación Normal (Z) porque n={n} (n ≥ 30) a pesar de no conocer σ."
            color_box = "success"
        else:
            v_critico = get_t_value(c, n - 1)
            dist_used = f"t-Student (df={n-1})"
            razon = f"Se utiliza la distribución t de Student porque σ es desconocida y n={n} (n < 30)."
            color_box = "warning"

    err = v_critico * (s / math.sqrt(n))
    li, ls = m - err, m + err

    st.markdown(f"""
        <div style="background: rgba(56, 189, 248, 0.05); border-left: 5px solid #38bdf8; padding: 15px; border-radius: 0 12px 12px 0; margin: 20px 0;">
            <strong style="color: #38bdf8; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.1em;">Lógica Estadística Aplicada:</strong><br>
            <span style="color: #cbd5e1; font-size: 1.1rem;">{razon}</span>
        </div>
    """, unsafe_allow_html=True)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Valor Crítico", f"{v_critico:.4f}")
    m2.metric("Margen de Error", f"{err:.4f}")
    m3.metric("Error Estándar", f"{(s/math.sqrt(n)):.4f}")
    
    st.markdown(f"""
        <div style="background: linear-gradient(90deg, rgba(34, 197, 94, 0.2), rgba(34, 197, 94, 0.05)); border: 1px solid rgba(34, 197, 94, 0.3); padding: 20px; border-radius: 16px; text-align: center; margin-top: 20px;">
            <h3 style="margin: 0; color: #4ade80;">Intervalo de Confianza</h3>
            <p style="font-size: 1.8rem; font-weight: 800; margin: 10px 0; font-family: 'JetBrains Mono', monospace; color: #f8fafc;">[{li:.4f}, {ls:.4f}]</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("📝 Registrar"): 
        registrar_calculo("Intervalo", {"i_n":n,"i_m":m,"i_s":s,"i_c":c, "i_sigma_know":sigma_conocida})
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
