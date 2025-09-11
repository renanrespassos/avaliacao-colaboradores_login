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

# ===================== Utils PDF/Gráficos =====================
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
    tokens, out = s.split(" "), []
    for t in tokens:
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
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2','4','6','8','10'])
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

# ===================== Perguntas por nível =====================
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
        modelo = {**comum,
            "Versatilidade": comum["Versatilidade"] + ["Demonstra abertura para aprender tarefas novas."],
            "Técnica": comum["Técnica"] + ["Aplica conceitos básicos sob supervisão."],
            "Norma 17025": comum["Norma 17025"] + ["Reconhece a importância de registros e rastreabilidade."],
        }
    elif nivel == "Assistente":
        modelo = {**comum,
            "Trabalho em Equipe": comum["Trabalho em Equipe"] + ["Colabora com colegas e passa turnos corretamente."],
            "Técnica": comum["Técnica"] + ["Opera instrumentos e segue ITs com pouca intervenção."],
            "Norma 17025": comum["Norma 17025"] + ["Registra não conformidades simples quando identifica desvios."],
        }
    elif nivel == "Analista":
        modelo = {**comum,
            "Trabalho em Equipe": comum["Trabalho em Equipe"] + ["Apoia tecnicamente colegas e padroniza procedimentos."],
            "Técnica": comum["Técnica"] + ["Define setups, interpreta resultados e resolve problemas recorrentes."],
            "Norma 17025": comum["Norma 17025"] + ["Mantém evidências, revisa registros e apoia auditorias internas."],
            "Olhar sistêmico": comum["Olhar sistêmico"] + ["Antecip​a impactos das mudanças de processo nos clientes."],
        }
    else:  # Especialista
        modelo = {**comum,
            "Trabalho em Equipe": comum["Trabalho em Equipe"] + ["Lidera frentes técnicas e treina o time."],
            "Técnica": comum["Técnica"] + ["Investiga causas-raiz e valida métodos/medições complexas."],
            "Norma 17025": comum["Norma 17025"] + ["Conduz auditorias internas e representa o laboratório em auditorias externas."],
            "Foco em Resultados": comum["Foco em Resultados"] + ["Define metas técnicas/qualidade e acompanha indicadores."],
        }
    return modelo

# ===================== APP =====================
st.set_page_config(page_title="Avaliação de Desempenho", layout="centered")
st.title(APP_TITLE)

# 1) Nível e modelo
nivel = st.selectbox("Nível do cargo (altera perguntas/skills padrão)",
                     ["Estagiário", "Assistente", "Analista", "Especialista"])
modelo_base = perguntas_padrao_por_nivel(nivel)
perguntas_cfg = modelo_base

# 2) Importar/Exportar modelo (JSON local)
st.markdown("### 🔄 Modelo de Perguntas (salvar/carregar)")
cimp, cexp = st.columns(2)

# Importar (carregar) JSON
with cimp:
    up = st.file_uploader("📥 Carregar modelo (JSON)", type=["json"], key="upload_model")
    if up is not None:
        try:
            data = json.loads(up.read().decode("utf-8"))
            if isinstance(data, dict) and "models" in data:
                modelo_nivel = data.get("models", {}).get(nivel)
                if modelo_nivel:
                    perguntas_cfg = modelo_nivel
                    st.success("Modelo do arquivo aplicado para o nível selecionado.")
            elif isinstance(data, dict):
                perguntas_cfg = data
                st.success("Modelo do arquivo aplicado.")
        except Exception as e:
            st.error(f"Não foi possível ler o JSON: {e}")

# Exportar (baixar) JSON
with cexp:
    export_payload = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models": {nivel: perguntas_cfg}
    }
    export_bytes = json.dumps(export_payload, ensure_ascii=False, indent=2).encode("utf-8")
    st.download_button("📤 Baixar meu modelo (JSON)", data=export_bytes,
                       file_name=f"modelo_perguntas_{nivel}.json", mime="application/json")

# 3) Edição opcional do modelo
editar = st.checkbox("Quero editar as perguntas", value=False)
if editar:
    with st.expander("🔧 Editar perguntas (opcional)"):
        categorias_txt = st.text_area("Categorias (separe por vírgula):",
                                      ",".join(perguntas_cfg.keys()), key=f"cats_{nivel}")
        categorias = [c.strip() for c in categorias_txt.split(",") if c.strip()]
        novo = {}
        for cat in categorias:
            default = "\n".join(perguntas_cfg.get(cat, []))
            txt = st.text_area(f"Perguntas para **{cat}** (1 por linha):", default, key=f"ta_{nivel}_{cat}")
            novo[cat] = [q.strip() for q in txt.split("\n") if q.strip()]
        perguntas_cfg = novo

# Info do colaborador
col1, col2 = st.columns(2)
with col1:
    colaborador = st.text_input("Nome do colaborador avaliado")
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

prev_file = st.file_uploader("📈 (Opcional) Envie o CSV da **última avaliação** para comparar evolução", type=["csv"])
prev_pdf  = st.file_uploader("📎 (Opcional) Anexar PDF da última avaliação (não entra no cálculo)", type=["pdf"])

# Coleta
notas, categorias, perguntas_list = [], [], []
obs_por_categoria = {}

st.header("Preencha a avaliação")
for categoria, qs in perguntas_cfg.items():
    st.subheader(categoria)
    for i, q in enumerate(qs):
        val = st.slider(q, 1, 10, 5, key=f"sl_{nivel}_{categoria}_{i}")
        notas.append(val); categorias.append(categoria); perguntas_list.append(q)
    obs = st.text_area(f"Observações sobre {categoria} (opcional):", key=f"obs_{nivel}_{categoria}")
    if obs.strip():
        obs_por_categoria[categoria] = obs.strip()

pontos_positivos = st.text_area("✅ Pontos positivos (opcional):")
oportunidades    = st.text_area("🔧 Oportunidades de melhorias (opcional):")

# ---------- PDF ----------
def gerar_pdf(nome_colaborador, nome_avaliador, data_avaliacao, nivel, df, media_atual,
              obs_categorias, pontos_pos, oportunidades, radar_buf, media_ant=None):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.set_left_margin(12); pdf.set_right_margin(12)
    font_name = pdf_set_unicode_font(pdf)
    pdf.add_page()

    # Cabeçalho
    pdf.set_font(font_name, "B", 16)
    pdf.cell(0, 10, "Relatório de Avaliação de Desempenho", ln=True, align="C"); pdf.ln(6)
    pdf.set_font(font_name, "", 12)
    pdf.cell(0, 8, f"Colaborador: {nome_colaborador}", ln=True)
    pdf.cell(0, 8, f"Avaliador: {nome_avaliador}", ln=True)
    pdf.cell(0, 8, f"Nível do cargo: {nivel}", ln=True)
    pdf.cell(0, 8, f"Data da avaliação: {data_avaliacao.strftime('%d/%m/%Y')}", ln=True)
    pdf.cell(0, 8, f"Média final: {df['Nota'].mean():.2f}", ln=True); pdf.ln(4)

    # Médias
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_font(font_name, "B", 12); pdf.cell(0, 8, "Média por Categoria:", ln=True)
    pdf.set_font(font_name, "", 11)
    for cat, val in media_atual.items():
        line = f"{cat}: {val:.2f}"
        if media_ant is not None and cat in media_ant.index and not pd.isna(media_ant[cat]):
            line = f"{line}  (Δ {val - media_ant[cat]:+.2f})"
        pdf.set_x(pdf.l_margin); pdf.multi_cell(page_w, 7, wrap_long_tokens(line))
    pdf.ln(4)

    # Radar
    img = Image.open(radar_buf); tmp_path = "radar_tmp.png"; img.save(tmp_path)
    pdf.image(tmp_path, x=45, w=120); pdf.ln(6)

    # Observações por categoria
    if obs_categorias:
        pdf.set_font(font_name, "B", 12); pdf.cell(0, 8, "Observações por Categoria:", ln=True)
        pdf.set_font(font_name, "", 11)
        for cat, texto in obs_categorias.items():
            pdf.set_x(pdf.l_margin); pdf.multi_cell(page_w, 6, wrap_long_tokens(f"{cat}:"))
            pdf.set_x(pdf.l_margin); pdf.multi_cell(page_w, 6, wrap_long_tokens(texto)); pdf.ln(1)

    # Pontos finais
    if (pontos_pos or "").strip():
        pdf.set_font(font_name, "B", 12); pdf.cell(0, 8, "Pontos positivos:", ln=True)
        pdf.set_font(font_name, "", 11)
        pdf.set_x(pdf.l_margin); pdf.multi_cell(page_w, 6, wrap_long_tokens(pontos_pos.strip())); pdf.ln(2)

    if (oportunidades or "").strip():
        pdf.set_font(font_name, "B", 12); pdf.cell(0, 8, "Oportunidades de melhorias:", ln=True)
        pdf.set_font(font_name, "", 11)
        pdf.set_x(pdf.l_margin); pdf.multi_cell(page_w, 6, wrap_long_tokens(oportunidades.strip())); pdf.ln(2)

    # Perguntas e notas
    pdf.set_font(font_name, "B", 12); pdf.cell(0, 8, "Perguntas e Notas:", ln=True)
    pdf.set_font(font_name, "", 10)
    for _, row in df.iterrows():
        txt = f"[{row['Categoria']}] {row['Pergunta']} - Nota: {row['Nota']}"
        pdf.set_x(pdf.l_margin); pdf.multi_cell(page_w, 6, wrap_long_tokens(txt))
    return pdf

# ---------- Geração ----------
if st.button("Gerar Relatório"):
    if not perguntas_cfg:
        st.warning("Configure ao menos uma categoria/pergunta para gerar o relatório.")
    else:
        df = pd.DataFrame({"Categoria": categorias, "Pergunta": perguntas_list, "Nota": notas})
        st.dataframe(df)

        media_atual = df.groupby("Categoria")["Nota"].mean().reindex(sorted(set(categorias)))
        media_ant = None
        try:
            if prev_file is not None:
                df_prev = pd.read_csv(prev_file)
                if set(["Categoria","Pergunta","Nota"]).issubset(df_prev.columns):
                    media_ant = df_prev.groupby("Categoria")["Nota"].mean().reindex(media_atual.index).astype(float)
                    st.subheader("Comparativo com última avaliação")
                    delta_df = pd.DataFrame({
                        "Categoria": media_atual.index,
                        "Média Anterior": media_ant.values,
                        "Média Atual": media_atual.values,
                        "Δ (Atual - Anterior)": media_atual.values - media_ant.values
                    })
                    st.dataframe(delta_df)
                    fig = plot_radar({"Atual": media_atual, "Anterior": media_ant}, "Radar comparativo (Atual x Anterior)")
                else:
                    st.warning("CSV inválido. Use o CSV exportado pelo app (colunas: Categoria, Pergunta, Nota).")
                    fig = plot_radar({"Atual": media_atual})
            else:
                fig = plot_radar({"Atual": media_atual})
        except Exception as e:
            st.warning(f"Não foi possível ler o CSV enviado: {e}")
            fig = plot_radar({"Atual": media_atual})

        st.subheader("Gráfico Radar"); st.pyplot(fig)
        st.subheader("Média por Categoria"); st.bar_chart(media_atual)
        st.write(f"**Média final do colaborador:** {df['Nota'].mean():.2f}")

        csv = df.to_csv(index=False).encode()
        st.download_button("Download do Relatório (CSV)", csv, "relatorio.csv")

        buf = BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", dpi=160); buf.seek(0)

        pdf = gerar_pdf(colaborador, avaliador, data_hoje, nivel,
                        df, media_atual, obs_por_categoria, pontos_positivos, oportunidades,
                        buf, media_ant=media_ant)
        result = pdf.output(dest="S")
        pdf_bytes = result if isinstance(result, (bytes, bytearray)) else result.encode("latin1")
        pdf_buf = BytesIO(pdf_bytes)

        nome_colab = sanitize_filename(colaborador)
        data_str = data_hoje.strftime("%Y-%m-%d")
        filename = f"relatorio_avaliacao_{nome_colab}_{nivel}_{data_str}.pdf"
        st.download_button("Baixar Relatório em PDF", pdf_buf, filename, mime="application/pdf")

        if prev_pdf is not None:
            st.info("PDF anterior anexado (somente referência; não entra no cálculo).")
            st.download_button("Baixar PDF anterior anexado",
                               prev_pdf.getvalue(),
                               file_name=f"avaliacao_anterior_{nome_colab}.pdf",
                               mime="application/pdf")

