# EDA Agent - Sistema Multi-Agente para Análise Exploratória de Dados

Sistema multi-agente genérico para Análise Exploratória de Dados (EDA) capaz de processar qualquer arquivo CSV e gerar insights, gráficos e conclusões através de consultas em linguagem natural.

## Funcionalidades

- **Processamento Genérico de CSV**: Manipula qualquer estrutura de arquivo CSV
- **Interface de Consulta em Linguagem Natural**: Aceita perguntas em linguagem natural
- **Geração Automatizada de Análises**: Gera código Python para análise de dados
- **Geração de Gráficos**: Cria representações visuais dos dados com Matplotlib e Plotly
- **Sistema de Memória Contextual**: Mantém contexto de conversa e histórico de análises
- **Múltiplos Agentes Especializados**: Agentes para análise de dados, detecção de padrões, anomalias e relacionamentos
- **Execução Segura de Código**: Sandbox para execução segura de código Python gerado

## Estrutura do Projeto

```
EDA_Agent_deploy/
├── app.py                          # Interface Streamlit
├── pyproject.toml                  # Configuração do projeto e dependências
├── .env.example                    # Exemplo de variáveis de ambiente
├── .gitignore                      # Arquivos ignorados pelo Git
├── README.md                       # Este arquivo
├── src/
│   ├── agents/                     # Agentes especializados
│   │   ├── anomaly_detector.py     # Detecção de anomalias
│   │   ├── code_generator.py       # Geração de código
│   │   ├── conclusion_generator.py # Geração de conclusões
│   │   ├── data_analyzer.py        # Análise de dados
│   │   ├── pattern_detector.py     # Detecção de padrões
│   │   └── relationship_analyzer.py # Análise de relacionamentos
│   ├── core/                       # Lógica principal
│   │   ├── code_executor.py        # Execução segura de código
│   │   ├── config.py               # Configurações do sistema
│   │   ├── csv_processor.py        # Processamento de CSV
│   │   ├── llm_manager.py          # Gerenciamento de LLMs
│   │   ├── logger.py               # Sistema de logs
│   │   ├── memory_manager.py       # Gerenciamento de memória
│   │   └── query_interpreter.py    # Interpretação de consultas
│   ├── graph/                      # Workflow LangGraph
│   │   ├── edges.py                # Definição de transições
│   │   ├── nodes.py                # Nós do workflow
│   │   ├── state.py                # Estado do grafo
│   │   └── workflow.py             # Orquestração do workflow
│   ├── models/                     # Modelos de dados
│   ├── tools/                      # Ferramentas de análise
│   ├── utils/                      # Utilitários
│   └── interface/                  # Componentes de interface
└── tests/                          # Testes automatizados
    ├── conftest.py                 # Configurações de testes
    ├── fixtures/                   # Dados de teste
    └── test_*.py                   # Suítes de testes
```

## Requisitos

- Python >= 3.9
- Chave de API do OpenAI
- Opcional: Chave do LangChain para tracing

## Instalação

### 1. Clone o repositório
```bash
git clone https://github.com/seu-usuario/eda-agent.git
cd eda-agent
```

### 2. Crie um ambiente virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instale as dependências
```bash
pip install -e .
```

### 4. Configure as variáveis de ambiente
```bash
cp .env.example .env
```

Edite o arquivo `.env` e adicione sua chave da API do OpenAI:
```env
OPENAI_API_KEY=sua_chave_aqui
```

## Uso

### Executar a aplicação

```bash
streamlit run app.py
```

A aplicação estará disponível em `http://localhost:8501`

### Como usar

1. Faça upload de um arquivo CSV
2. Faça perguntas em linguagem natural sobre seus dados
3. Visualize análises, gráficos e insights gerados automaticamente

### Exemplos de consultas

- "Quais são as estatísticas descritivas dos dados?"
- "Existem outliers nos dados?"
- "Mostre a correlação entre as variáveis"
- "Há valores faltantes?"
- "Analise a distribuição das variáveis numéricas"

## Desenvolvimento

### Instalar dependências de desenvolvimento
```bash
pip install -e ".[dev]"
```

### Executar testes
```bash
pytest
```

### Executar testes com cobertura
```bash
pytest --cov=src tests/
```

### Formatação de código
```bash
black src/ tests/
isort src/ tests/
```

### Linting
```bash
flake8 src/ tests/
mypy src/
```

## Deploy

### Streamlit Cloud

1. Faça push do código para o GitHub
2. Acesse [Streamlit Cloud](https://streamlit.io/cloud)
3. Conecte seu repositório
4. Configure as variáveis de ambiente no dashboard
5. Deploy!

### Docker (opcional)

```bash
# Build
docker build -t eda-agent .

# Run
docker run -p 8501:8501 --env-file .env eda-agent
```

## Configuração

Todas as configurações estão em `src/core/config.py` e podem ser sobrescritas via variáveis de ambiente:

- `OPENAI_API_KEY`: Chave da API OpenAI (obrigatório)
- `LOG_LEVEL`: Nível de log (padrão: INFO)
- `EXECUTION_TIMEOUT`: Timeout de execução em segundos (padrão: 30)
- `MAX_UPLOAD_SIZE_MB`: Tamanho máximo de upload (padrão: 200)

## Segurança

- Execução de código em sandbox isolado
- Validação de entrada de dados
- Sanitização de código Python gerado
- Restrição de importações perigosas
- Logs detalhados de todas as operações

## Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## Licença

MIT License

## Suporte

Para questões e suporte, abra uma issue no GitHub.