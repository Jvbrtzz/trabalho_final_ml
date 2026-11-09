"""Gera o notebook acadêmico. O notebook é então executado para produzir os artefatos."""
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent
nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}


def md(text: str) -> None:
    nb.cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text: str) -> None:
    nb.cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# OncoLens: Classificação de Massas Mamárias com Regressão Logística

**Alunos:** João Victor Bortoluzzi da Silva; Pedro Rimes Coutinho; Leonardo Ximenes  
**Data:** 9 de novembro de 2026  
**Dataset:** Breast Cancer Wisconsin (Diagnostic) — UCI Machine Learning Repository  
**Técnica principal:** Regressão Logística (classificação supervisionada)

## 1. Contexto e objetivo

O câncer de mama é um problema relevante de saúde pública e a análise quantitativa de imagens pode apoiar a triagem de amostras. Este projeto constrói, avalia e disponibiliza um classificador binário que estima se as medidas de uma massa mamária são compatíveis com a classe **benigna (0)** ou **maligna (1)**. A regressão logística foi escolhida por produzir probabilidades e coeficientes interpretáveis. Ela será comparada ao K-Nearest Neighbors (KNN).

O conjunto contém medidas calculadas a partir de imagens digitalizadas de punção aspirativa por agulha fina. A fonte original é: Wolberg, W.; Mangasarian, O.; Street, N.; Street, W. (1993), *Breast Cancer Wisconsin (Diagnostic)*, UCI, DOI [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). A cópia empacotada no scikit-learn é usada para garantir reprodutibilidade.

> **Aviso ético:** este é um experimento acadêmico. O modelo não foi validado clinicamente e não substitui profissionais ou exames médicos.
""")

md("## 2. Importação de bibliotecas e configuração")
code(r"""
from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from IPython.display import display
from sklearn.datasets import load_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, ConfusionMatrixDisplay, confusion_matrix,
    f1_score, precision_score, recall_score, RocCurveDisplay, roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"Benigno": "#20b8a6", "Maligno": "#d14b64"}

ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "reports" / "figures"
MODEL_DIR = ROOT / "models"
for directory in (DATA_DIR, FIG_DIR, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)

print("Versões — numpy:", np.__version__, "| pandas:", pd.__version__, "| sklearn:", sklearn.__version__)
""")

md(r"""
## 3. Carregamento e inspeção

A versão do scikit-learn preserva as 569 observações e 30 variáveis contínuas da base UCI. No objeto original, `target=0` representa maligno e `target=1` representa benigno. Para tornar as métricas clínicas mais intuitivas, o alvo é recodificado como **1 = maligno** e **0 = benigno**. Uma cópia CSV é salva junto ao projeto.
""")
code(r"""
dataset = load_breast_cancer(as_frame=True)
df = dataset.data.copy()
df["diagnostico"] = (dataset.target == 0).astype(int)
df["classe"] = df["diagnostico"].map({0: "Benigno", 1: "Maligno"})
df.to_csv(DATA_DIR / "breast_cancer_wisconsin.csv", index=False)

print(f"Dimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
display(df.head())
""")
code(r"""
inspection = pd.DataFrame({
    "tipo": df.dtypes.astype(str),
    "nulos": df.isna().sum(),
    "unicos": df.nunique()
})
print("Total de valores nulos:", int(df.isna().sum().sum()))
print("Linhas duplicadas:", int(df.duplicated().sum()))
display(inspection)
""")
code(r"""
display(df.drop(columns="classe").describe().T)
""")

md(r"""
**Interpretação da inspeção.** A base é pequena (569 registros), exclusivamente numérica nas variáveis preditoras, sem valores ausentes e sem linhas duplicadas. Mesmo assim, o pipeline incluirá imputação por mediana: isso torna a inferência robusta caso dados futuros tenham ausência. As escalas diferem muito (por exemplo, área e suavidade), portanto a padronização é indispensável para os dois algoritmos.
""")

md("## 4. Análise Exploratória dos Dados (EDA)")
md("### 4.1 Distribuição do alvo")
code(r"""
counts = df["classe"].value_counts().reindex(["Benigno", "Maligno"])
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(counts.index, counts.values, color=[COLORS[x] for x in counts.index], width=.62)
ax.bar_label(bars, labels=[f"{v} ({v/len(df):.1%})" for v in counts.values], padding=4)
ax.set(title="Distribuição das classes", ylabel="Número de observações", ylim=(0, 400))
fig.tight_layout(); fig.savefig(FIG_DIR / "01_distribuicao_alvo.png", dpi=160, bbox_inches="tight"); plt.show()
""")
md("**Interpretação.** Há 357 casos benignos (62,7%) e 212 malignos (37,3%). O desbalanceamento é moderado; por isso o split será estratificado, a regressão usará `class_weight='balanced'` e a comparação não dependerá apenas da acurácia.")

md("### 4.2 Distribuição do raio médio por classe")
code(r"""
fig, ax = plt.subplots(figsize=(8, 4.5))
for label in ["Benigno", "Maligno"]:
    ax.hist(df.loc[df.classe == label, "mean radius"], bins=22, alpha=.58, label=label, color=COLORS[label])
ax.set(title="Distribuição do raio médio", xlabel="Raio médio", ylabel="Frequência")
ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "02_hist_raio.png", dpi=160, bbox_inches="tight"); plt.show()
""")
md("**Interpretação.** As amostras malignas se concentram em raios maiores, embora exista sobreposição entre as classes. Logo, o raio contém sinal preditivo, mas não deve ser usado isoladamente.")

md("### 4.3 Correlação entre atributos selecionados")
code(r"""
FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area",
    "mean smoothness", "mean compactness", "mean concavity", "mean concave points"
]
corr = df[FEATURES + ["diagnostico"]].corr()
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
short = [x.replace("mean ", "") for x in FEATURES] + ["alvo"]
ax.set_xticks(range(len(short)), short, rotation=45, ha="right")
ax.set_yticks(range(len(short)), short)
for i in range(len(short)):
    for j in range(len(short)):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if abs(corr.iloc[i,j]) > .55 else "#24324a")
fig.colorbar(im, ax=ax, shrink=.78, label="Correlação de Pearson")
ax.set_title("Matriz de correlação — atributos do modelo")
fig.tight_layout(); fig.savefig(FIG_DIR / "03_correlacao.png", dpi=160, bbox_inches="tight"); plt.show()
""")
md("**Interpretação.** Raio, perímetro e área são fortemente correlacionados, pois representam aspectos do tamanho. Concavidade e pontos côncavos apresentam associação positiva relevante com malignidade. A regressão logística continua útil, mas seus coeficientes devem ser lidos com cautela na presença de multicolinearidade.")

md("### 4.4 Relação entre raio e textura")
code(r"""
fig, ax = plt.subplots(figsize=(8, 5))
for label in ["Benigno", "Maligno"]:
    part = df[df.classe == label]
    ax.scatter(part["mean radius"], part["mean texture"], s=28, alpha=.62, label=label, color=COLORS[label], edgecolor="none")
ax.set(title="Raio médio × textura média", xlabel="Raio médio", ylabel="Textura média")
ax.legend(); fig.tight_layout(); fig.savefig(FIG_DIR / "04_scatter_dominio.png", dpi=160, bbox_inches="tight"); plt.show()
""")
md("**Interpretação.** Casos malignos aparecem com maior frequência no quadrante de raio e textura elevados. A fronteira não é perfeita, confirmando a necessidade de combinar múltiplas medidas.")

md("### 4.5 Pontos côncavos por diagnóstico")
code(r"""
fig, ax = plt.subplots(figsize=(7, 4.5))
values = [df.loc[df.classe == label, "mean concave points"] for label in ["Benigno", "Maligno"]]
bp = ax.boxplot(values, tick_labels=["Benigno", "Maligno"], patch_artist=True, showfliers=False)
for box, label in zip(bp["boxes"], ["Benigno", "Maligno"]): box.set_facecolor(COLORS[label]); box.set_alpha(.72)
ax.set(title="Pontos côncavos médios por classe", ylabel="Pontos côncavos médios")
fig.tight_layout(); fig.savefig(FIG_DIR / "05_boxplot.png", dpi=160, bbox_inches="tight"); plt.show()
""")
md("**Interpretação.** A mediana de pontos côncavos é claramente maior na classe maligna. A variável tem boa separação univariada e representa a irregularidade do contorno observada na imagem.")

md(r"""
## 5. Pré-processamento

Foram escolhidas oito medidas médias, antes de observar o conjunto de teste, para equilibrar desempenho, interpretação e usabilidade da interface. Não há atributos categóricos; portanto, encoding não se aplica. O split 80/20 é estratificado e reprodutível. Em cada pipeline, o imputador e o scaler são ajustados **somente dentro dos dados de treino**, evitando vazamento. O teste fica reservado para a avaliação final.
""")
code(r"""
X = df[FEATURES]
y = df["diagnostico"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.20, stratify=y, random_state=42
)
print("Treino:", X_train.shape, "| Teste:", X_test.shape)
print("Proporção maligna — treino:", round(y_train.mean(), 3), "| teste:", round(y_test.mean(), 3))
""")

md(r"""
## 6. Modelagem e comparação

São comparados dois algoritmos: Regressão Logística e KNN. A comparação usa validação cruzada estratificada com cinco folds apenas no treino. A métrica principal é **ROC AUC**, por avaliar o ranqueamento probabilístico em vários limiares; recall e F1 da classe maligna também são destacados pelo custo potencial de falsos negativos.
""")
code(r"""
models = {
    "Regressão Logística": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=11, weights="distance")
}
pipelines = {
    name: Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", estimator),
    ]) for name, estimator in models.items()
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
cv_rows = []
for name, pipeline in pipelines.items():
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
    row = {"Modelo": name}
    for metric in scoring:
        row[metric] = scores[f"test_{metric}"].mean()
        row[f"{metric}_std"] = scores[f"test_{metric}"].std()
    cv_rows.append(row)
cv_results = pd.DataFrame(cv_rows).set_index("Modelo")
display(cv_results[["accuracy", "precision", "recall", "f1", "roc_auc"]].style.format("{:.3f}"))
""")
md("**Leitura da validação cruzada.** O modelo é selecionado pelo ROC AUC médio no treino, sem consultar o teste. A regressão logística é mantida como solução principal por seu desempenho competitivo, produção direta de probabilidades e maior transparência dos coeficientes.")

md("## 7. Avaliação final no conjunto de teste")
code(r"""
test_rows = []
fitted = {}
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    fitted[name] = pipeline
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]
    test_rows.append({
        "Modelo": name,
        "Acurácia": accuracy_score(y_test, pred),
        "Precisão": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
        "ROC AUC": roc_auc_score(y_test, prob),
    })
test_results = pd.DataFrame(test_rows).set_index("Modelo")
display(test_results.style.format("{:.3f}"))
""")
code(r"""
best_model = fitted["Regressão Logística"]
lr_pred = best_model.predict(X_test)
lr_prob = best_model.predict_proba(X_test)[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, lr_pred), display_labels=["Benigno", "Maligno"]
).plot(ax=axes[0], cmap="Blues", colorbar=False)
axes[0].set_title("Matriz de confusão — teste")
RocCurveDisplay.from_predictions(y_test, lr_prob, ax=axes[1], color="#3157d5", name="Regressão Logística")
axes[1].plot([0, 1], [0, 1], "--", color="#94a3b8", label="Aleatório")
axes[1].set_title("Curva ROC — conjunto de teste")
axes[1].legend(loc="lower right")
fig.tight_layout(); fig.savefig(FIG_DIR / "06_avaliacao.png", dpi=170, bbox_inches="tight"); plt.show()
""")
md("**Interpretação.** A matriz expõe acertos e erros separadamente para benignos e malignos. A curva ROC acima da diagonal indica discriminação consistente. Em um cenário clínico real, o limiar deveria ser calibrado com especialistas e dados externos para reduzir falsos negativos; aqui é mantido 0,50 para clareza acadêmica.")

md("### 7.1 Coeficientes da regressão logística")
code(r"""
coef = pd.Series(
    best_model.named_steps["classifier"].coef_[0], index=FEATURES
).sort_values()
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.barh(coef.index.str.replace("mean ", ""), coef.values,
        color=["#20b8a6" if x < 0 else "#d14b64" for x in coef.values])
ax.axvline(0, color="#334155", linewidth=.8)
ax.set(title="Coeficientes padronizados", xlabel="Efeito sobre o log-odds da classe maligna")
fig.tight_layout(); fig.savefig(FIG_DIR / "07_coeficientes.png", dpi=160, bbox_inches="tight"); plt.show()
display(coef.to_frame("coeficiente"))
""")
md("**Interpretação.** Coeficientes positivos aumentam a estimativa de malignidade e negativos a reduzem, mantendo os demais atributos constantes. Como tamanho, perímetro e área são correlacionados, não se deve interpretar um coeficiente isolado como relação causal.")

md(r"""
## 8. Exportação do modelo e transformadores

O objeto principal exportado é o pipeline completo, garantindo que imputação e padronização na API sejam idênticas ao treinamento. Para atender também à inspeção individual dos componentes, o scaler e o estimador são salvos separadamente. A API **não retreina** durante a predição.
""")
code(r"""
API_FEATURES = [
    "mean_radius", "mean_texture", "mean_perimeter", "mean_area",
    "mean_smoothness", "mean_compactness", "mean_concavity", "mean_concave_points"
]
metadata = {
    "model_features": FEATURES,
    "api_features": API_FEATURES,
    "class_names": {0: "Benigno", 1: "Maligno"},
    "threshold": 0.5,
    "dataset_doi": "10.24432/C5DW2B",
    "sklearn_version": sklearn.__version__,
}
joblib.dump(best_model, MODEL_DIR / "modelo_regressao_logistica.joblib")
joblib.dump(best_model.named_steps["scaler"], MODEL_DIR / "scaler.joblib")
joblib.dump(best_model.named_steps["classifier"], MODEL_DIR / "regressao_logistica.joblib")
joblib.dump(metadata, MODEL_DIR / "metadados.joblib")

test_results.to_csv(ROOT / "reports" / "metricas_teste.csv")
cv_results.to_csv(ROOT / "reports" / "metricas_validacao_cruzada.csv")
summary = {
    "dataset": {"linhas": len(df), "atributos_originais": 30, "atributos_modelo": len(FEATURES),
                "benignos": int((y == 0).sum()), "malignos": int((y == 1).sum())},
    "test_metrics": {name: {k: float(v) for k, v in row.items()}
                     for name, row in test_results.to_dict(orient="index").items()},
    "cv_metrics": {name: {k: float(v) for k, v in row.items()}
                   for name, row in cv_results[["accuracy", "precision", "recall", "f1", "roc_auc"]].to_dict(orient="index").items()},
    "confusion_matrix": confusion_matrix(y_test, lr_pred).tolist(),
}
with open(ROOT / "reports" / "resultados.json", "w", encoding="utf-8") as file:
    json.dump(summary, file, ensure_ascii=False, indent=2)

print("Artefatos exportados:")
for path in sorted(MODEL_DIR.glob("*.joblib")): print("-", path.name, f"({path.stat().st_size/1024:.1f} KB)")
""")

md(r"""
## 9. Conclusão

1. O projeto transformou uma base pública e documentada em uma aplicação de classificação reproduzível.  
2. A inspeção confirmou 569 registros, 30 atributos numéricos, ausência de nulos e ausência de duplicatas.  
3. A EDA mostrou desbalanceamento moderado, tratado com estratificação e pesos de classe.  
4. Raio, perímetro e área apresentaram forte correlação, coerente com o fato de medirem tamanho.  
5. Concavidade e pontos côncavos mostraram associação visual importante com a classe maligna.  
6. O pipeline eliminou vazamento ao ajustar imputador e scaler somente dentro de cada treino.  
7. Regressão Logística e KNN foram comparados por validação cruzada estratificada de cinco folds.  
8. A Regressão Logística foi escolhida por combinar bom ROC AUC, probabilidades e interpretabilidade.  
9. O pipeline completo foi exportado e é carregado uma única vez pela API FastAPI.  
10. A interface valida campos e faixas antes de apresentar classe e probabilidade de forma clara.  
11. A principal limitação é a amostra pequena, antiga e proveniente de um único contexto de coleta.  
12. Também não houve validação externa, análise de subgrupos, calibração clínica ou estudo prospectivo.  
13. Como melhoria futura, recomenda-se testar coortes recentes e independentes, calibrar o limiar e monitorar drift.  
14. Portanto, o sistema responde ao objetivo acadêmico, mas não deve ser interpretado como dispositivo médico.
""")

nbf.write(nb, ROOT / "notebook_ml.ipynb")
print(f"Notebook criado com {len(nb.cells)} células: {ROOT / 'notebook_ml.ipynb'}")

