# Chat Bot com IA para whatsapp🤖

Aqui vai uma pequna explicação para que você possa montar seu próprio RAG personalizado:

**RAG (Retrieval-Augmented Generation)📕**

RAG é uma *técnica que melhora a geração de texto por IA* (como respostas) ao permitir que o modelo acesse e utilize informações externas (documentos, bases de conhecimento) antes de formular a resposta. **Em vez de se basear apenas no treinamento, ele recupera dados relevantes para garantir respostas mais precisas e atualizadas.**

**Chunks✏️**

Chunks são **pequenas partes ou pedaços em que um texto extenso** (documentos, artigos) é dividido. Essa fragmentação facilita o processamento, armazenamento e, principalmente, a **recuperação eficiente de informações específicas pelo sistema.**

**Overlap📖**

Overlap é a *sobreposição intencional de texto entre chunks adjacentes*. É uma técnica para **garantir que nenhuma informação crucial seja cortada ou perdida** na fronteira de um chunk, mantendo a continuidade e o contexto do texto original.

**Vector Store✍️**

A Vector Store (ou Armazenamento de Vetores) *é um tipo de banco de dados especializado que armazena dados em forma de vetores.* Ela é **essencial para o RAG**, **permitindo que o sistema realize buscas ultra-rápidas por similaridade semântica (por significado) entre a consulta do usuário e os documentos armazenados.**


**Embeddings📒**

Embeddings são *representações numéricas (vetores) que um modelo de linguagem gera a partir de palavras, frases ou chunks.* Eles capturam o significado semântico do texto: **quanto mais próximos os vetores estiverem no espaço, mais semelhantes são os seus significados** (ex: "carro" e "automóvel"). *São a base para a busca na Vector Store.*


Se você **não tem** Docker instalado na sua máquina ***instale-o imediatamente!!***

Alguns comandos úteis para você:

``` docker ps -a ```  para listar todos os seus containers

``` docker system prune -a ``` para deletar todos os recursos (TODOS MESMO, containers, imagens, volumes, redes) **TOME CUIDADO QUANDO USAR ESSE COMANDO**⚠️

``` docker exec -it <container> /bin/bash ``` para acessar o shell de algum container

``` docker-compose up --build ``` para 'buildar' e subir todos os serviços com o docker-compose

``` docker-compose up ``` para **apenas** subir todos os serviços com o docker-compose


É possível também buildar e subir apenas um serviço para o docker, por exemplo:
Neste projeto temos duas coisas que vamos buildar no docker - o serviço local do WAHA (API) e a nossa API, você pode então usar o comando ```docker-compose up --build waha``` ou ```docker-compose up --build api```. 

* Instale as dependências com ```pip install -r requirements.txt```

* Construa seu serviço Waha com ```docker-compose up --build waha```

* Construa a api do chat bot com ```docker-compose up --build api```

Acesse o Waha no seu navegador e entre no dashboard com o login e senha que aparecerá no seu terminal. Configure seu número de whatsapp e teste. 
Caso tenha dificuldades com a configuração leia a documentação oficial, ela poderá te ajudar! 😁

**Link da documentação oficial: https://waha.devlike.pro/docs/how-to/config/**

Após testar e ver que está tudo funcionando, você pode personalizar seu RAG e se divertir com seu novo chat bot com IA!!


----
**Esse chat bot utiliza o modelo de linguagem 'llama-3.1-70b-versatile' (Groq)** 🔗🤖

**O arquivo ```requirements.txt```foi escrito manualmente de forma resumida por conta dos problemas de versão entre as bibliotecas Langchain. Se preferir e quiser modificar o requirements para um arquivo mais robusto, após instalar as dependências use o comando ```pip freeze > requirements.txt``` para gerar o arquivo extenso.**

##  📬 Contribuições

Contribuições são super bem-vindas! 🤝
Basta abrir uma issue ou um pull request.

##  💖 Feito com ❤️ por Rayssa Santos
---
💖desenvolvido com base nos vídeos do canal **[PycodeBR](https://www.youtube.com/@pycodebr)**




