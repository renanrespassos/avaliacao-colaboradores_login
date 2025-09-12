import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from PIL import Image
from datetime import datetime
import os, re, json

APP_TITLE = "Avaliação de Desempenho do Colaborador"
FONT_PATH = os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")

# ----------------- Utils -----------------
def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>| ]', '_', (name or "").strip()) or "colaborador"

def pdf_set_unicode_font(pdf: FPDF) -> str:
    try:
        if os.path.exists(FONT_PATH):
            pdf.add_font("DejaVu", "", FONT_PATH, uni=True)
            pdf.add_font("DejaVu", "B", FONT_PATH, uni=True)
            return "DejaVu"
    except Exception:
        pass
    return "Arial"

def wrap_long_tokens(s: str, max_len: int = 60) -> str:
    s = str(s).replace("\t", " ").replace("\u00A0", " ").strip()
    out = []
    for t in s.split(" "):
        if len(t) > max_len:
            out.extend([t[i:i+max_len] for i in range(0, len(t), max_len)])
        else:
            out.append(t)
    return " ".join(out)

def plot_radar(series_dict, title="Desempenho por Categoria"):
    labels = list(next(iter(series_dict.values())).index)
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles_c = angles + angles[:1]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.set_title(title, y=1.08)
    ax.set_xticks(angles); ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([2, 4, 6, 8, 10]); ax.set_yticklabels(['2','4','6','8','10'])
    ax.grid(True)
    for lbl, ser in series_dict.items():
        vals = list(ser.values); vals_c = vals + vals[:1]
        ax.plot(angles_c, vals_c, linewidth=2, label=lbl)
        ax.fill(angles_c, vals_c, alpha=0.15)
        if lbl.lower().startswith("atual"):
            for ang, v in zip(angles, vals):
                ax.annotate(f"{v:.1f}", xy=(ang, v), xytext=(0,8),
                            textcoords="offset points", ha="center", va="bottom", fontsize=9)
    if len(series_dict) > 1:
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.10))
    return fig

# ----------------- Perguntas por nível -----------------
def perguntas_padrao_por_nivel(nivel: str) -> dict:
    comum = {
        "Versatilidade": ["Contribui com a equipe para cumprir metas e prazos."],
        "Relacionamento": ["Mantém relacionamento respeitoso e colaborativo."],
        "Olhar sistêmico": ["Compreende como seu trabalho impacta clientes e áreas parceiras."],
        "Trabalho em Equipe": ["Compartilha informações e pede ajuda quando necessário."],
        "Responsabilidade": ["Cumpre prazos e segue orientações e políticas."],
        "Foco em Resultados": ["Entrega as atividades previstas com qualidade."],
        "Organização": ["Organiza o próprio trabalho e prioriza demandas."],
        "Norma 17025": ["Segue instruções de trabalho e padrões aplicáveis."],
        "Técnica": ["Aplica corretamente procedimentos e registra evidências."],
    }
    if nivel == "Estagiário":
        return {
            **comum,
            "Versatilidade": comum["Versatilidade"] + ["Demonstra abertura para aprender tarefas novas."],
            "Técnica": comum["Técnica"] + ["Aplica conceitos básicos sob supervisão."],
            "Norma 17025": comum["Norma 17025"] + ["Reconhece a importância de registros e rastreabilidade."],
        }
    if nivel == "Assistente":
        return {
            **comum,
            "Trabalho em Equipe": comum["Trabalho em Equipe"] + ["Colabora com colegas e passa turnos corretamente."],
            "Técnica": comum["Técnica"] + ["Opera instrumentos e segue ITs com pouca intervenção."],
            "Norma 17025": comum["Norma 17025"] + ["Registra não conformidades simples quando identifica desvios."],
        }
    if nivel == "Analista":
        return {
            **comum,
            "Trabalho em Equipe": comum["Trabalho em Equipe"] + ["Apoia tecnicamente colegas e padroniza procedimentos."],
            "Técnica": comum["Técnica"] + ["Define setups, interpreta resultados e resolve problemas recorrentes."],
            "Norma 17025": comum["Norma 17025"] + ["Mantém evidências, revisa registros e apoia auditorias internas."],
            "Olhar sistêmico": comum["Olhar sistêmico"] + ["Antecip​a impactos das mudanças de processo nos clientes."],
        }
    # Especialista
    return {
        **comum,
        "Trabalho em Equipe": comum["Trabalho em Equipe"] + ["Lidera frentes técnicas e treina o time."],
        "Técnica": comum["Técnica"] + ["Investiga causas-raiz e valida métodos/medições complexas."],
        "Norma 17025": comum["Norma 17025"] + ["Conduz auditorias internas e representa o laboratório em auditorias externas."],
        "Foco em Resultados": comum["Foco em Resultados"] + ["Define metas técnicas/qualidade e acompanha indicadores."],
    }

# ----------------- APP -----------------
st.set_page_config(page_title="Avaliação de Desempenho", layout="centered")
st.title(APP_TITLE)

# estado global
if "modelos" not in st.session_state:
    st.session_state["modelos"] = {}
if "edit_open" not in st.session_state:
    st.session_state["edit_open"] = False
if "edit_pending_close" not in st.session_state:
    st.session_state["edit_pending_close"] = False

nivel = st.selectbox("Nível do cargo", ["Estagiário", "Assistente", "Analista", "Especialista"])
if nivel not in st.session_state["modelos"]:
    st.session_state["modelos"][nivel] = perguntas_padrao_por_nivel(nivel)

# --- Importar/Exportar JSON ---
st.markdown("### 🔄 Modelo de Perguntas (salvar/carregar)")
cimp, cexp = st.columns(2)

with cimp:
    up = st.file_uploader("📥 Carregar modelo (JSON)", type=["json"], key=f"upload_{nivel}")
    if up is not None:
        try:
            data = json.loads(up.read().decode("utf-8"))
            if isinstance(data, dict) and "models" in data:
                modelo_nivel = data.get("models", {}).get(nivel)
                if modelo_nivel:
                    st.session_state["modelos"][nivel] = modelo_nivel
                    st.success("Modelo aplicado do arquivo.")
        except Exception as e:
            st.error(f"Erro no JSON: {e}")

with cexp:
    export_payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {nivel: st.session_state['modelos'][nivel]}
    }
    st.download_button("📤 Baixar meu modelo", 
                       json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name=f"modelo_perguntas_{nivel}.json", mime="application/json")

# --- Editor com salvar/fechar ---
if st.session_state["edit_pending_close"]:
    st.session_state["edit_open"] = False
    st.session_state["edit_open_chk"] = False
    st.session_state["edit_pending_close"] = False

edit_chk = st.checkbox("Quero editar as perguntas",
                       key="edit_open_chk",
                       value=st.session_state["edit_open"])
st.session_state["edit_open"] = bool(edit_chk)

if st.session_state["edit_open"]:
    with st.expander("🔧 Editar perguntas", expanded=True):
        categorias_txt = st.text_area("Categorias (separe por vírgula):",
                                      ",".join(st.session_state["modelos"][nivel].keys()),
                                      key=f"cats_{nivel}")
        categorias = [c.strip() for c in categorias_txt.split(",") if c.strip()]
        edits = {}
        for cat in categorias:
            default = "\n".join(st.session_state["modelos"][nivel].get(cat, []))
            txt = st.text_area(f"Perguntas para **{cat}** (1 por linha):",
                               default, key=f"ta_{nivel}_{cat}")
            edits[cat] = [q.strip() for q in txt.split("\n") if q.strip()]
        c1, c2 = st.columns(2)
        if c1.button("💾 Salvar alterações", use_container_width=True):
            st.session_state["modelos"][nivel] = edits
            st.session_state["edit_pending_close"] = True
            st.success("Alterações salvas!")
            st.rerun()
        if c2.button("Cancelar", use_container_width=True):
            st.session_state["edit_pending_close"] = True
            st.info("Edição cancelada.")
            st.rerun()

# --- Infos colaborador ---
col1, col2 = st.columns(2)
with col1:
    colaborador = st.text_input("Nome do colaborador")
with col2:
    avaliador = st.text_input("Nome do avaliador (opcional)")
data_hoje = st.date_input("Data da avaliação", datetime.today())

st.markdown("""
**Responda de 1 a 10 conforme a legenda:**

| Nota | Interpretação  |
|:---:|-----------------|
| 1-2 | **Nunca**       |
| 3-4 | **Raramente**   |
| 5-6 | **Às vezes**    |
| 7-8 | **Frequentemente** |
| 9-10| **Sempre**      |
""")

prev_file = st.file_uploader("📈 (Opcional) CSV da última avaliação", type=["csv"])
prev_pdf  = st.file_uploader("📎 (Opcional) PDF anterior", type=["pdf"])

# --- Coleta ---
notas, categorias, perguntas_list = [], [], []
obs_por_categoria = {}
for categoria, qs in st.session_state["modelos"][nivel].items():
    st.subheader(categoria)
    for i, q in enumerate(qs):
        val = st.slider(q, 1, 10, 5, key=f"sl_{nivel}_{categoria}_{i}")
        notas.append(val); categorias.append(categoria); perguntas_list.append(q)
    obs = st.text_area(f"Observações sobre {categoria} (opcional):", key=f"obs_{nivel}_{categoria}")
    if obs.strip(): obs_por_categoria[categoria] = obs.strip()

pontos_positivos = st.text_area("✅ Pontos positivos (opcional):")
oportunidades    = st.text_area("🔧 Oportunidades de melhorias (opcional):")

# --- PDF ---
def gerar_pdf(nome_colaborador, nome_avaliador, data_avaliacao, nivel, df, media_atual,
              obs_categorias, pontos_pos, oportunidades, radar_buf, media_ant=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.set_left_margin(12); pdf.set_right_margin(12)
    font_name = pdf_set_unicode_font(pdf)
    pdf.add_page()
    pdf.set_font(font_name, "B", 16)
    pdf.cell(0, 10, "Relatório de Avaliação de Desempenho", ln=True, align="C"); pdf.ln(6)
    pdf.set_font(font_name, "", 12)
    pdf.cell(0, 8, f"Colaborador: {nome_colaborador}", ln=True)
    pdf.cell(0, 8, f"Avaliador: {nome_avaliador}", ln=True)
    pdf.cell(0, 8, f"Nível: {nivel}", ln=True)
    pdf.cell(0, 8, f"Data: {data_avaliacao.strftime('%d/%m/%Y')}", ln=True)
    pdf.cell(0, 8, f"Média final: {df['Nota'].mean():.2f}", ln=True); pdf.ln(4)
    return pdf

# --- Geração ---
if st.button("Gerar Relatório"):
    df = pd.DataFrame({"Categoria": categorias, "Pergunta": perguntas_list, "Nota": notas})
    st.dataframe(df)
    media_atual = df.groupby("Categoria")["Nota"].mean().reindex(sorted(set(categorias)))
    fig = plot_radar({"Atual": media_atual})
    st.pyplot(fig)

