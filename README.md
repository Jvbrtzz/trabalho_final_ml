# OncoLens — classificação com Regressão Logística

Projeto acadêmico completo de Machine Learning para classificar amostras de massas mamárias como benignas ou malignas. O modelo principal é uma **Regressão Logística**, comparada ao KNN por validação cruzada.

> Uso exclusivamente educacional. O resultado não substitui avaliação, exame ou diagnóstico médico.

## Entregáveis

- `notebook_ml.ipynb`: experimento completo e já executado;
- `relatorio_projeto_ml.pdf`: relatório ilustrado com mais de 5 páginas;
- `app.py`: ponto de entrada mínimo da API FastAPI;
- `oncolens/`: backend organizado em camadas;
- `static/`: HTML, CSS e JavaScript separados por responsabilidade;
- `models/modelo_regressao_logistica.joblib`: pipeline pronto para inferência;
- `models/scaler.joblib` e `models/regressao_logistica.joblib`: componentes também exportados separadamente;
- `data/breast_cancer_wisconsin.csv`: cópia local dos dados usados;
- `tests/test_app.py`: testes automatizados da API.

## Como executar

No PowerShell, a partir desta pasta:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app:app --reload
```

Acesse `http://127.0.0.1:8000`. A documentação automática da API fica em `http://127.0.0.1:8000/docs`.

Para abrir o notebook:

```powershell
jupyter notebook notebook_ml.ipynb
```

Para testar:

```powershell
python -m unittest discover -s tests -v
```

## Arquitetura

```text
trabalho_final_ml/
├── app.py                         # Composition root ASGI
├── oncolens/
│   ├── application.py             # Application factory e injeção de dependências
│   ├── config.py                  # Caminhos e configurações
│   ├── api/routes/
│   │   ├── health.py              # Monitoramento da API
│   │   └── prediction.py          # Adaptação HTTP do caso de uso
│   ├── domain/prediction.py       # Resultado independente de frameworks
│   ├── infrastructure/
│   │   └── model_repository.py    # Carregamento dos arquivos joblib
│   ├── schemas/prediction.py      # Contratos Pydantic
│   ├── services/
│   │   └── prediction_service.py  # Regra de predição
│   └── web/routes.py              # Entrega da página principal
├── static/
│   ├── index.html                 # Marcação e conteúdo
│   ├── css/styles.css             # Apresentação visual
│   └── js/
│       ├── app.js                 # Coordenação da interface
│       ├── api.js                 # Comunicação com a API
│       ├── config.js              # Campos e dados de demonstração
│       ├── validation.js          # Validação do formulário
│       └── ui.js                  # Renderização dos estados da tela
├── models/                        # Artefatos treinados
└── tests/                         # Testes de API e unidade
```

O fluxo de dependências aponta para dentro: as rotas conhecem o serviço, o serviço conhece somente o domínio e o repositório fica isolado na infraestrutura. O `app.py` não contém regras de negócio.

## Endpoints

- `GET /`: interface web;
- `GET /api/v1/health`: disponibilidade e versão;
- `POST /api/v1/predictions`: classificação de uma amostra;
- `GET /docs`: documentação interativa OpenAPI.

## Dataset

Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository, DOI: [10.24432/C5DW2B](https://doi.org/10.24432/C5DW2B). São 569 observações, 30 atributos contínuos e duas classes. A cópia distribuída pelo scikit-learn foi usada para garantir reprodutibilidade.

Antes de entregar, substitua `[PREENCHER NOME]` no notebook e no relatório pelo nome do(a) aluno(a).

