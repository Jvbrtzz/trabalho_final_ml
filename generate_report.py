"""Gera um relatório PDF paginado a partir dos resultados reais do notebook."""
from pathlib import Path
import json
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Circle
import pandas as pd


ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SHOTS = REPORTS / "screenshots"
OUTPUT = ROOT / "relatorio_projeto_ml.pdf"

with open(REPORTS / "resultados.json", encoding="utf-8") as file:
    results = json.load(file)
df = pd.read_csv(ROOT / "data" / "breast_cancer_wisconsin.csv")

NAVY = "#17233b"
BLUE = "#3157d5"
TEAL = "#20b8a6"
MUTED = "#64748b"
LIGHT = "#f4f7fb"
RED = "#d14b64"

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": NAVY, "axes.titlecolor": NAVY})


def base_page(pdf: PdfPages, title: str, section: str, page: int):
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((.055, .925), .035, .035, boxstyle="round,pad=.008,rounding_size=.01",
                                transform=ax.transAxes, facecolor=BLUE, edgecolor="none"))
    ax.text(.0725, .942, "◎", ha="center", va="center", color="white", fontsize=12, fontweight="bold")
    ax.text(.102, .943, "ONCOLENS", fontsize=9, fontweight="bold", va="center", color=BLUE)
    ax.text(.055, .895, section.upper(), fontsize=8, fontweight="bold", color=TEAL)
    ax.text(.055, .855, title, fontsize=22, fontweight="bold", va="top")
    ax.plot([.055, .945], [.82, .82], color="#dce4ef", lw=1, transform=ax.transAxes)
    ax.text(.055, .028, "Projeto acadêmico • Regressão Logística • Wisconsin Diagnostic", fontsize=7.5, color=MUTED)
    ax.text(.945, .028, str(page), fontsize=8, color=MUTED, ha="right")
    return fig, ax


def paragraph(ax, text, x, y, width=96, size=10, color=NAVY, line=0.023, weight="normal"):
    wrapped = textwrap.wrap(text, width=width, break_long_words=False)
    ax.text(x, y, "\n".join(wrapped), transform=ax.transAxes, va="top", fontsize=size,
            color=color, linespacing=1.42, fontweight=weight)
    return y - line * max(1, len(wrapped)) * 1.42


def image_ax(fig, path, box):
    ax = fig.add_axes(box); ax.axis("off")
    ax.imshow(plt.imread(path))
    return ax


with PdfPages(OUTPUT) as pdf:
    # 1 — Capa
    fig = plt.figure(figsize=(8.27, 11.69), facecolor=NAVY)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.add_patch(Circle((.85, .86), .24, transform=ax.transAxes, color=TEAL, alpha=.13))
    ax.add_patch(Circle((.08, .12), .30, transform=ax.transAxes, color=BLUE, alpha=.20))
    ax.text(.08, .89, "◎  ONCOLENS", color="#9ee7df", fontsize=12, fontweight="bold")
    ax.text(.08, .69, "CLASSIFICAÇÃO DE\nMASSAS MAMÁRIAS", color="white", fontsize=33,
            fontweight="bold", va="top", linespacing=1.08)
    ax.text(.08, .50, "Um projeto completo de Machine Learning\ncom Regressão Logística", color="#cbd5e1",
            fontsize=16, va="top", linespacing=1.5)
    ax.plot([.08, .34], [.39, .39], color=TEAL, lw=4, transform=ax.transAxes)
    ax.text(.08, .31, "ALUNO(A)", color="#8292ab", fontsize=8, fontweight="bold")
    ax.text(.08, .275, "[PREENCHER NOME]", color="white", fontsize=14, fontweight="bold")
    ax.text(.08, .20, "DATA", color="#8292ab", fontsize=8, fontweight="bold")
    ax.text(.08, .165, "26 de outubro de 2026", color="white", fontsize=12)
    ax.text(.08, .07, "Breast Cancer Wisconsin (Diagnostic) • UCI Machine Learning Repository", color="#8292ab", fontsize=8)
    pdf.savefig(fig); plt.close(fig)

    # 2 — Introdução
    fig, ax = base_page(pdf, "Introdução", "01 • Contexto", 2)
    y = .785
    y = paragraph(ax, "A análise de imagens de punção aspirativa por agulha fina produz medidas quantitativas da forma dos núcleos celulares. Este trabalho investiga se oito dessas medidas permitem distinguir padrões benignos e malignos de maneira reproduzível.", .07, y)
    y -= .02
    ax.add_patch(FancyBboxPatch((.07, y-.15), .86, .15, boxstyle="round,pad=.018,rounding_size=.015",
                                transform=ax.transAxes, facecolor="#eef2ff", edgecolor="#cfd8ff"))
    ax.text(.095, y-.035, "OBJETIVO", transform=ax.transAxes, color=BLUE, fontsize=8, fontweight="bold")
    paragraph(ax, "Construir um classificador binário, comparar algoritmos, avaliar o modelo em dados separados e disponibilizar a Regressão Logística em uma interface web com validação de entradas.", .095, y-.07, width=78, size=11)
    y -= .20
    ax.text(.07, y, "Por que Regressão Logística?", transform=ax.transAxes, fontsize=14, fontweight="bold")
    y -= .04
    for title, desc in [
        ("Probabilidade", "produz uma estimativa entre 0 e 1, fácil de comunicar."),
        ("Interpretabilidade", "permite inspecionar o sinal dos coeficientes padronizados."),
        ("Baseline sólido", "é simples, rápido e adequado à classificação binária."),
        ("Implantação", "o pipeline salvo pode ser carregado diretamente pela API."),
    ]:
        ax.add_patch(Circle((.087, y-.012), .012, transform=ax.transAxes, color=TEAL))
        ax.text(.087, y-.012, "✓", transform=ax.transAxes, color="white", fontsize=7, ha="center", va="center")
        ax.text(.115, y, title, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")
        ax.text(.255, y, desc, transform=ax.transAxes, fontsize=9.5, color=MUTED, va="top")
        y -= .065
    ax.add_patch(FancyBboxPatch((.07, .105), .86, .09, boxstyle="round,pad=.016,rounding_size=.012",
                                transform=ax.transAxes, facecolor="#fff9df", edgecolor="#edd889"))
    paragraph(ax, "Responsabilidade: o sistema é exclusivamente educacional, não foi validado clinicamente e não substitui avaliação, exame ou diagnóstico realizado por profissional de saúde.", .09, .17, width=80, size=9, color="#6b5b18")
    pdf.savefig(fig); plt.close(fig)

    # 3 — Dataset
    fig, ax = base_page(pdf, "Dataset", "02 • Dados", 3)
    ax.text(.07, .785, "Breast Cancer Wisconsin (Diagnostic)", transform=ax.transAxes, fontsize=14, fontweight="bold")
    paragraph(ax, "Fonte: UCI Machine Learning Repository, Wolberg et al. (1993), DOI 10.24432/C5DW2B. A cópia do scikit-learn foi usada para reprodutibilidade.", .07, .752, width=90, size=9.5, color=MUTED)
    cards = [("569", "registros"), ("30", "atributos originais"), ("357", "benignos"), ("212", "malignos")]
    for i, (value, label) in enumerate(cards):
        x = .07 + i * .22
        ax.add_patch(FancyBboxPatch((x, .63), .19, .085, boxstyle="round,pad=.012,rounding_size=.01",
                                    transform=ax.transAxes, facecolor=LIGHT, edgecolor="#dce4ef"))
        ax.text(x+.02, .685, value, transform=ax.transAxes, fontsize=18, fontweight="bold", color=BLUE)
        ax.text(x+.02, .652, label, transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.text(.07, .585, "Atributos usados pelo modelo", transform=ax.transAxes, fontsize=12, fontweight="bold")
    descriptions = [
        ("Raio, perímetro e área", "medidas de tamanho do núcleo."),
        ("Textura", "variação dos tons de cinza."),
        ("Suavidade", "variação local dos comprimentos do raio."),
        ("Compacidade", "relação entre perímetro e área."),
        ("Concavidade", "severidade de porções côncavas do contorno."),
        ("Pontos côncavos", "quantidade de porções côncavas."),
    ]
    y = .545
    for name, desc in descriptions:
        ax.text(.085, y, "•", transform=ax.transAxes, color=TEAL, fontsize=13, va="top")
        ax.text(.11, y, name, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
        ax.text(.36, y, desc, transform=ax.transAxes, fontsize=9, color=MUTED, va="top")
        y -= .044
    sample = df[["mean radius", "mean texture", "mean area", "mean concavity", "diagnostico"]].head(4).round(4)
    tax = fig.add_axes([.07, .135, .86, .20]); tax.axis("off")
    table = tax.table(cellText=sample.values, colLabels=["raio", "textura", "área", "concavidade", "alvo"],
                      cellLoc="center", loc="center")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.7)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dce4ef")
        cell.set_facecolor("#eef2ff" if row == 0 else "white")
        if row == 0: cell.set_text_props(weight="bold", color=BLUE)
    ax.text(.07, .345, "Amostra dos dados", transform=ax.transAxes, fontsize=11, fontweight="bold")
    pdf.savefig(fig); plt.close(fig)

    # 4 — EDA
    fig, ax = base_page(pdf, "Análise exploratória", "03 • EDA", 4)
    image_ax(fig, FIGURES / "01_distribuicao_alvo.png", [.07, .52, .40, .27])
    image_ax(fig, FIGURES / "02_hist_raio.png", [.53, .52, .40, .27])
    image_ax(fig, FIGURES / "03_correlacao.png", [.07, .16, .40, .31])
    image_ax(fig, FIGURES / "04_scatter_dominio.png", [.53, .16, .40, .31])
    ax.text(.07, .485, "• Há desbalanceamento moderado: 62,7% benignos e 37,3% malignos.", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.text(.07, .455, "• Casos malignos tendem a apresentar raios maiores, mas existe sobreposição.", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.text(.07, .125, "• Medidas de tamanho são correlacionadas; concavidade e pontos côncavos se associam ao alvo.", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    ax.text(.07, .095, "• Nenhuma variável isolada separa perfeitamente as classes, justificando um modelo multivariado.", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    pdf.savefig(fig); plt.close(fig)

    # 5 — Metodologia
    fig, ax = base_page(pdf, "Metodologia", "04 • Pipeline", 5)
    steps = [
        ("1", "Split estratificado", "80% treino / 20% teste\nrandom_state = 42"),
        ("2", "Imputação", "Mediana ajustada\nsomente no treino"),
        ("3", "Padronização", "StandardScaler dentro\ndo pipeline"),
        ("4", "Comparação", "Regressão Logística × KNN\nCV estratificada, 5 folds"),
        ("5", "Teste final", "Acurácia, precisão, recall,\nF1, matriz e ROC AUC"),
    ]
    y = .75
    for n, title, desc in steps:
        ax.add_patch(Circle((.11, y), .027, transform=ax.transAxes, color=BLUE))
        ax.text(.11, y, n, transform=ax.transAxes, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.text(.16, y+.012, title, transform=ax.transAxes, fontsize=11, fontweight="bold", va="center")
        ax.text(.43, y+.012, desc, transform=ax.transAxes, fontsize=9, color=MUTED, va="center", linespacing=1.4)
        if n != "5": ax.plot([.11, .11], [y-.075, y-.035], transform=ax.transAxes, color="#c7d2fe", lw=3)
        y -= .13
    ax.add_patch(FancyBboxPatch((.07, .09), .86, .12, boxstyle="round,pad=.015,rounding_size=.012",
                                transform=ax.transAxes, facecolor="#effcf9", edgecolor="#a9e6dc"))
    ax.text(.095, .17, "CONTROLE DE VAZAMENTO", transform=ax.transAxes, color="#087c6e", fontsize=8, fontweight="bold")
    paragraph(ax, "Imputador e scaler permanecem dentro de cada pipeline. Assim, cada fold aprende os parâmetros apenas com sua parcela de treino, e o conjunto de teste não participa da escolha do modelo.", .095, .14, width=78, size=9, color="#315c58")
    pdf.savefig(fig); plt.close(fig)

    # 6 — Resultados
    fig, ax = base_page(pdf, "Resultados", "05 • Avaliação", 6)
    lr = results["test_metrics"]["Regressão Logística"]
    knn = results["test_metrics"]["KNN"]
    ax.text(.07, .78, "Métricas no conjunto de teste", transform=ax.transAxes, fontsize=12, fontweight="bold")
    metrics = ["Acurácia", "Precisão", "Recall", "F1-Score", "ROC AUC"]
    table_data = [["Regressão Logística"] + [f"{lr[m]:.3f}" for m in metrics],
                  ["KNN"] + [f"{knn[m]:.3f}" for m in metrics]]
    tax = fig.add_axes([.07, .64, .86, .12]); tax.axis("off")
    table = tax.table(cellText=table_data, colLabels=["Modelo"] + metrics, cellLoc="center", loc="center",
                      colWidths=[.25, .15, .15, .15, .15, .15])
    table.auto_set_font_size(False); table.set_fontsize(7.5); table.scale(1, 1.9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#dce4ef"); cell.set_facecolor("#eef2ff" if row == 0 else "white")
        if row == 0: cell.set_text_props(weight="bold", color=BLUE)
        if row == 1 and col == 0: cell.set_text_props(weight="bold")
    image_ax(fig, FIGURES / "06_avaliacao.png", [.08, .27, .84, .31])
    cm = results["confusion_matrix"]
    ax.add_patch(FancyBboxPatch((.07, .105), .86, .10, boxstyle="round,pad=.015,rounding_size=.012",
                                transform=ax.transAxes, facecolor=LIGHT, edgecolor="#dce4ef"))
    paragraph(ax, f"A regressão logística identificou {cm[1][1]} dos {cm[1][0] + cm[1][1]} casos malignos no teste (recall {lr['Recall']:.1%}), com {cm[0][1]} falsos positivos e {cm[1][0]} falsos negativos. A seleção foi guiada pelo ROC AUC médio na validação cruzada do treino ({results['cv_metrics']['Regressão Logística']['roc_auc']:.3f}), antes de consultar o teste.", .095, .18, width=80, size=9)
    pdf.savefig(fig); plt.close(fig)

    # 7 — Interface principal e sucesso
    fig, ax = base_page(pdf, "Interface gráfica", "06 • Aplicação", 7)
    ax.text(.07, .785, "Tela principal", transform=ax.transAxes, fontsize=11, fontweight="bold")
    image_ax(fig, SHOTS / "01_tela_inicial.png", [.07, .47, .86, .30])
    ax.text(.07, .435, "Caso de sucesso: predição válida", transform=ax.transAxes, fontsize=11, fontweight="bold")
    image_ax(fig, SHOTS / "04_refatoracao_validada.png", [.07, .11, .86, .30])
    ax.text(.07, .075, "A API retorna classe, probabilidade da classe maligna, modelo e aviso de uso responsável.", transform=ax.transAxes, fontsize=8.5, color=MUTED)
    pdf.savefig(fig); plt.close(fig)

    # 8 — Erro e execução
    fig, ax = base_page(pdf, "Validação e execução", "07 • Uso", 8)
    ax.text(.07, .785, "Caso negativo: valor fora da faixa plausível", transform=ax.transAxes, fontsize=11, fontweight="bold")
    image_ax(fig, SHOTS / "03_entrada_invalida.png", [.07, .46, .86, .30])
    paragraph(ax, "O navegador valida campos vazios, tipos e limites. A API repete a validação com Pydantic; portanto, uma chamada direta também recebe HTTP 422 quando os dados são inválidos.", .07, .425, width=90, size=9.5, color=MUTED)
    ax.text(.07, .335, "Como executar no PowerShell", transform=ax.transAxes, fontsize=12, fontweight="bold")
    commands = ["python -m venv .venv", ".\\.venv\\Scripts\\Activate.ps1", "pip install -r requirements.txt", "uvicorn app:app --reload"]
    ax.add_patch(FancyBboxPatch((.07, .16), .86, .14, boxstyle="round,pad=.018,rounding_size=.012",
                                transform=ax.transAxes, facecolor=NAVY, edgecolor="none"))
    ax.text(.095, .275, "\n".join(commands), transform=ax.transAxes, fontsize=9.5, family="monospace", color="#d7e1f0", va="top", linespacing=1.55)
    ax.text(.07, .12, "Depois, acesse http://127.0.0.1:8000. A documentação da API fica em /docs.", transform=ax.transAxes, fontsize=9, color=MUTED)
    pdf.savefig(fig); plt.close(fig)

    # 9 — Conclusão
    fig, ax = base_page(pdf, "Conclusão", "08 • Encerramento", 9)
    y = .785
    conclusions = [
        "O ciclo completo foi realizado: inspeção, EDA, pré-processamento, comparação, teste, exportação e interface.",
        "A regressão logística apresentou ROC AUC de {:.3f} e recall de {:.3f} no teste.".format(lr["ROC AUC"], lr["Recall"]),
        "O pipeline evita vazamento e garante que a mesma padronização seja aplicada na API.",
        "A interface torna o resultado compreensível e trata entradas inválidas em duas camadas.",
        "A amostra é pequena, antiga e não representa toda a diversidade de populações e equipamentos.",
        "Não houve validação externa, calibração clínica nem auditoria de desempenho por subgrupos.",
        "Trabalhos futuros devem usar coortes independentes, calibrar o limiar e monitorar drift.",
    ]
    for i, item in enumerate(conclusions, 1):
        ax.add_patch(Circle((.095, y-.01), .016, transform=ax.transAxes, color=TEAL if i < 5 else RED, alpha=.9))
        ax.text(.095, y-.01, str(i), transform=ax.transAxes, color="white", fontsize=7, ha="center", va="center", fontweight="bold")
        paragraph(ax, item, .13, y, width=78, size=9.5)
        y -= .077
    ax.text(.07, .225, "Referências e arquivos", transform=ax.transAxes, fontsize=12, fontweight="bold")
    refs = [
        "Wolberg, W. et al. (1993). Breast Cancer Wisconsin (Diagnostic). UCI. DOI: 10.24432/C5DW2B.",
        "scikit-learn. load_breast_cancer e documentação de Pipeline, LogisticRegression e métricas.",
        "Artefatos: notebook_ml.ipynb, app.py, static/index.html e models/*.joblib.",
    ]
    y = .19
    for ref in refs:
        y = paragraph(ax, "• " + ref, .08, y, width=90, size=8.5, color=MUTED, line=.019) - .008
    pdf.savefig(fig); plt.close(fig)

print(f"Relatório gerado: {OUTPUT} ({OUTPUT.stat().st_size / 1024:.1f} KB)")

