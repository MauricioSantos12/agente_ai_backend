// backend/server.js

const express = require("express");
const axios = require("axios");
const cors = require("cors");
const fs = require("fs");
const pdf = require("pdf-parse");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const { ChatOpenAI } = require("@langchain/openai");
const {
  ChatPromptTemplate,
  MessagesPlaceholder,
} = require("@langchain/core/prompts");
const { HumanMessage, AIMessage } = require("@langchain/core/messages");
const {
  createOpenAIFunctionsAgent,
  AgentExecutor,
} = require("langchain/agents");
const {
  TavilySearchResults,
} = require("@langchain/community/tools/tavily_search");
const { createRetrieverTool } = require("langchain/tools/retriever");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const {
  CheerioWebBaseLoader,
} = require("langchain/document_loaders/web/cheerio");
const { MemoryVectorStore } = require("langchain/vectorstores/memory");
const { OpenAIEmbeddings } = require("@langchain/openai");
const { Document } = require("langchain/document");

require("dotenv").config();

const app = express();
const PORT = 8080;

app.use(cors());
app.use(express.json()); // Para parsear JSON

const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// Función para extraer texto de un PDF
const extractPDFContent = async (filePath) => {
  const dataBuffer = fs.readFileSync(filePath);
  const data = await pdf(dataBuffer);
  return data.text;
};

// Función para cargar y extraer contenido de PDFs
const loadPdfDataUtel = async () => {
  const pdfFiles = [
    {
      name: "Utel",
      filePath: "./pdfs/Qué es Utel.pdf",
    },
  ];

  const pdfData = {};

  for (const pdfFile of pdfFiles) {
    const content = await extractPDFContent(pdfFile.filePath);
    pdfData[pdfFile.name] = content;
  }

  return pdfData;
};

const getRetrieverFromPDFs = async () => {
  // Cargar el contenido de los PDFs
  const pdfData = await loadPdfDataUtel();
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 700,
    chunkOverlap: 100,
  });

  // Dividir cada documento en fragmentos
  const splitDocs = [];
  for (const [name, content] of Object.entries(pdfData)) {
    // const doc = { text: content, metadata: { name } };
    const doc = new Document({
      pageContent: content,
      metadata: { name },
    });
    const chunks = await splitter.splitDocuments([doc]);
    splitDocs.push(...chunks);
  }
  const embeddings = new OpenAIEmbeddings();

  // Crear el vector store desde los fragmentos divididos
  const vectorStore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );
  const retriever = vectorStore.asRetriever({ k: 2 });
  return retriever;
};

const getRetrieverFromWebPage = async () => {
  const loader = new CheerioWebBaseLoader("https://utel.edu.mx/sobre-utel");
  const docs = await loader.load();

  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 2000,
    chunkOverlap: 100,
  });
  const splitDocs = await splitter.splitDocuments(docs);

  const embeddings = new OpenAIEmbeddings();

  const vectorstore = await MemoryVectorStore.fromDocuments(
    splitDocs,
    embeddings
  );

  const retriever = vectorstore.asRetriever({ k: 2 });
  return retriever;
};

app.get("/", (req, res) => {
  res.send("The server is ready");
});

app.post("/api/general-chat", async (req, res) => {
  const { message, chatHistory } = req.body;
  if (!message) {
    return res.status(400).send("Por favor, proporciona un message.");
  }
  const searchTool = new TavilySearchResults();

  const tools = [searchTool];

  // Instantiate the model
  const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 1,
  });

  // Prompt Template
  const prompt = ChatPromptTemplate.fromMessages([
    ("system", "Eres un agente experto y poderoso"),
    new MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
    new MessagesPlaceholder("agent_scratchpad"),
  ]);

  const agent = await createOpenAIFunctionsAgent({
    llm: model,
    prompt,
    tools,
  });

  // Create the executor
  const agentExecutor = new AgentExecutor({
    agent,
    tools,
  });

  try {
    const response = await agentExecutor.invoke({
      input: `${message}. Dame la respuesta con emojis y de manera amigable. Dámela en html para poder ingresarlo en un dangerouslySetInnerHTML, no agregues la etiqueta html. Etiquetas b, p, h2`,
      chat_history: chatHistory,
    });
    chatHistory.push(new HumanMessage(message));
    chatHistory.push(new AIMessage(response.output));
    res.status(200).json({
      responseModel: response.output,
      chatHistory: chatHistory,
    });
  } catch (error) {
    if (error.response) {
      console.error("Error al llamar a Gemini API:", error);
      return res.status(500).json({ error: "Error de la API de Gemini." });
    }
    console.error("Error:", error.message);
    res.status(500).json({ error: "Error en el servidor." });
  }
});

app.post("/api/general-chat-pdf", async (req, res) => {
  const { message, chatHistory } = req.body;
  if (!message) {
    return res.status(400).send("Por favor, proporciona un message.");
  }

  try {
    // Instantiate the model
    const model = new ChatOpenAI({
      modelName: "gpt-3.5-turbo",
      temperature: 0,
    });

    // Prompt Template
    const prompt = ChatPromptTemplate.fromMessages([
      ("system",
      "Eres un agente experto y poderoso de Utel universidad. NO respondas preguntas que no tienen relación con Utel"),
      // new MessagesPlaceholder("chat_history"),
      ("human", "{input}"),
      new MessagesPlaceholder("agent_scratchpad"),
    ]);

    // Tools
    const retrieverUtelPdf = await getRetrieverFromPDFs();
    const retrieverTool = createRetrieverTool(retrieverUtelPdf, {
      name: "utel_search",
      description: "Use this tool to search information and answer questions",
    });

    const tools = [retrieverTool];
    const agent = await createOpenAIFunctionsAgent({
      llm: model,
      prompt,
      tools,
    });

    // Create the executor
    const agentExecutor = new AgentExecutor({
      agent,
      tools,
    });

    const response = await agentExecutor.invoke({
      input: `${message}.`,
      chat_history: chatHistory,
    });
    chatHistory.push(new HumanMessage(message));
    chatHistory.push(new AIMessage(response.output));
    res.status(200).json({
      responseModel: response.output,
      chatHistory: chatHistory,
    });
  } catch (error) {
    if (error.response) {
      console.error("Error al llamar a OPENAI:", error);
      return res.status(500).json({ error: "Error en OPENAI." });
    }
    console.error("Error:", error.message);
    res.status(500).json({ error: "Error en el servidor." });
  }
});

app.post("/api/get-questions", async (req, res) => {
  const { country, ageRange, areas } = req.body;
  if (!areas) {
    return res.status(400).send("Por favor, proporciona una lista de areas.");
  }
  const searchTool = new TavilySearchResults();

  const tools = [searchTool];

  // Instantiate the model
  const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo",
    temperature: 1,
  });

  // Prompt Template
  const prompt = ChatPromptTemplate.fromMessages([
    // ("system", "You are a helpful assistant."),
    ("human", "{input}"),
    new MessagesPlaceholder("agent_scratchpad"),
  ]);

  const agent = await createOpenAIFunctionsAgent({
    llm: model,
    prompt,
    tools,
  });

  // Create the executor
  const agentExecutor = new AgentExecutor({
    agent,
    tools,
  });

  try {
    const response = await agentExecutor.invoke({
      input: `
      Hola estoy realizando un test vocacional y quiero que me des quince (15) preguntas claves para saber que deberia estudiar una persona, estas preguntas deben un formato de opcion multiple y algunas en escala de calificación numérica de muy poco a siempre, tambien usa en alguna pregunta la escala de likert sin mencionar la escala y que la persona tenga que escoger una opción.
      Las preguntas tienen que estar dentro del contexto del país: ${country} y el rango de edad de: ${ageRange}. 
      Ten en cuenta que la persona no sabe que estudiar, trata de dejar preguntas abiertas pero concisas.
      El formato de la pregunta es la siguiente:
        Acá inician las preguntas y respuestas: 
        - Pregunta: La pregunta que generas
        - Opciones: - a) Respuesta a - b) Respuesta b - c) Respuesta c - d) Respuesta d
        - Pregunta: La pregunta que generas
        - Opciones: - a) Respuesta a - b) Respuesta b - c) Respuesta c - d) Respuesta d
      Responde solo las preguntas y respuestas en el formato indicado
      `,
    });
    res.status(200).json({
      responseModel: response.output,
    });
  } catch (error) {
    if (error.response) {
      console.error("Error al llamar a Gemini API:", error);
      return res.status(500).json({ error: "Error de la API de Gemini." });
    }
    console.error("Error:", error.message);
    res.status(500).json({ error: "Error en el servidor." });
  }
});

app.post("/api/get-final-recomedation", async (req, res) => {
  const { name, country, textoConcatenado, ageRange, email } = req.body;
  if (!textoConcatenado) {
    return res.status(400).send("Por favor, proporciona un textoConcatenado.");
  }

  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

  const chat = model.startChat({
    history: [
      {
        role: "user",
        parts: [{ text: "Hola" }],
      },
      {
        role: "model",
        parts: [
          {
            text: "Hola, soy un recomendador de carreras de acuerdo a lo que me puedas indicar",
          },
        ],
      },
      {
        role: "user",
        parts: [{ text: "Muchas gracias" }],
      },
      {
        role: "model",
        parts: [
          {
            text: "Con gusto. Cuéntame tus gustos y te puedo asesorar",
          },
        ],
      },
    ],
  });

  try {
    let result = await chat.sendMessage(
      `
      Mi nombre es: ${name}. Vivo en ${country}, y tengo un rango de edad de: ${ageRange}.
      Dame algunas carreras que me sirvan de acuerdo a estas preguntas y respuestas: ${textoConcatenado}.
      También, necesito que la respuesta sea personalizada. Poniendo mi nombre, el país donde vivo, como me puede aportar esa área y carreras según mi rango de edad y algunas empresas que trabajan en esa área. 
      Dame la respuesta como un asesor de UTEL UNIVERSIDAD. Tiene que ser concisa. La respuesta la necesito para que genere motivación.
      Dame la respuesta en html para ponerlo en un dangerouslySetInnerHTML, no agregues la etiqueta html. Dame el texto con etiquetas h1, h2, p y b. 
      Dámelo con la siguiente estructura:
        1. Saluda a la persona y resalta su nombre en una etiqueta <b>. Así ¡Hola <b>{nombre} </b>! 👋.
        2. Una descripción breve de utel y como se ajusta con los datos que te proporcioné.
        3. En una cuadrícula de 2 columnas por 2 filas agrega las carreras en modo de cards para desktop con la siguiente configuración: repeat(auto-fit, minmax(350px, 1fr)).
          Para mobile debe ser una sola columna.
          Cada card debe contener lo siguiente:
            Debe ir dentro de un cuadro de color #e5e7eb y un padding y borde adecuado.
            Debe tener flex y flex-column, Debe tener un padding y margin adecuado y un border radius de 8px.
            Cada card debe tener un h3 con el título de la carrera, en negrilla. Un p para la desciprción y el listado enumerado de las empresas. Estas también deben ir con un flex y un direction column.
          4. Recuerda dar una respuesta para motivar a la persona a estudiar una nueva carrera. Añade un poco de emojis y resalta las empresas con una negrilla.
          5. El texto antes de resaltar las empresas debe ser sin negrilla y debe decir: "Empresas en las puedes trabajar: "
          6. Agrega mínimo 4 opciones en la cuadrícula mencionada.
      Todo el texto debe tener un padding entre el mismo para que no sea vea junto
       `
    );
    res.json(result.response.text());
  } catch (error) {
    if (error.response) {
      console.error("Error al llamar a la API:", error);
      return res.status(500).json({ error: "Error de la API." });
    }
    console.error("Error:", error.message);
    res.status(500).json({ error: "Error en el servidor." });
  }
});

app.post("/api/asesor", async (req, res) => {
  const { message, chatHistory } = req.body;
  if (!message) {
    return res.status(400).send("Por favor, proporciona un message.");
  }

  try {
    // Instantiate the model
    const model = new ChatOpenAI({
      modelName: "gpt-3.5-turbo",
      temperature: 1,
    });

    // Prompt Template
    const prompt = ChatPromptTemplate.fromMessages([
      ("system",
      ` Eres un poderoso asesor de Utel universidad que recomienda carreras. 
        Realiza 15 preguntas una por otra. 
        Al final de las 1 preguntas recomienda 3 carreras posibles.
        NECEISTO QUE: Al inicio de la respuesta final agrega: *****"Respuesta final*****"`),
      new MessagesPlaceholder("chat_history"),
      ("human", "{input}"),
      new MessagesPlaceholder("agent_scratchpad"),
    ]);

    // Tools
    const retrieverUtelPdf = await getRetrieverFromPDFs();
    const retrieverTool = createRetrieverTool(retrieverUtelPdf, {
      name: "utel_search",
      description: "Use this tool to search information and answer questions",
    });
    const tools = [retrieverTool];
    const agent = await createOpenAIFunctionsAgent({
      llm: model,
      prompt,
      tools,
    });

    // Create the executor
    const agentExecutor = new AgentExecutor({
      agent,
      tools,
    });

    const response = await agentExecutor.invoke({
      input: `${message}`,
      chat_history: chatHistory,
    });
    chatHistory.push(new HumanMessage(message));
    chatHistory.push(new AIMessage(response.output));
    res.status(200).json({
      responseModel: response.output,
      chatHistory: chatHistory,
    });
  } catch (error) {
    if (error.response) {
      console.error("Error al llamar a OPENAI:", error);
      return res.status(500).json({ error: "Error en OPENAI." });
    }
    console.error("Error:", error.message);
    res.status(500).json({ error: "Error en el servidor." });
  }
});

app.post("/api/get-final-asesor-recomedation", async (req, res) => {
  const { finalResponse } = req.body;
  if (!finalResponse) {
    return res.status(400).send("Por favor, proporciona un finalResponse.");
  }

  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

  const chat = model.startChat({
    history: [],
  });

  try {
    let result = await chat.sendMessage(
      `
      Busca algunas carreras que me sirvan de acuerdo a este texto: ${finalResponse}.
      Dame la respuesta como un asesor de UTEL UNIVERSIDAD. Tiene que ser concisa. La respuesta la necesito para que genere motivación.
      Dame la respuesta en html para ponerlo en un dangerouslySetInnerHTML, no agregues la etiqueta html. Dame el texto con etiquetas h1, h2, p y b. 
      Las carreras las renderizas con la siguiente estructura:
        1. Todo debe estar en un div de color blanco
        2. Una descripción breve de utel y como se ajusta con los datos que te proporcioné.
        3. En una cuadrícula de 2 columnas por 2 filas agrega las carreras en modo de "cards". Para desktop con la siguiente configuración: repeat(auto-fit, minmax(250px, 1fr)).
          Para mobile debe ser una sola columna.
          Cada card debe contener lo siguiente:
            Debe ir dentro de un cuadro de color #e5e7eb y un padding y borde de 6px.
            Debe tener flex y flex-column, Debe tener un padding y margin adecuado y un border radius de 8px.
            Cada card debe tener un h3 con el título de la carrera, en negrilla. Un p para la desciprción y el listado enumerado de las empresas. Estas también deben ir con un flex y un direction column.
          4. Recuerda dar una respuesta para motivar a la persona a estudiar una nueva carrera. Añade un poco de emojis y resalta las empresas con una negrilla.
          5. El texto antes de resaltar las empresas debe ser sin negrilla y debe decir: "Empresas en las puedes trabajar: "
          6. Agrega mínimo 4 opciones en la cuadrícula mencionada.
      Todo el texto debe tener un padding entre el mismo para que no sea vea junto
       `
    );
    res.json(result.response.text());
  } catch (error) {
    if (error.response) {
      console.error("Error al llamar a la API:", error);
      return res.status(500).json({ error: "Error de la API." });
    }
    console.error("Error:", error.message);
    res.status(500).json({ error: "Error en el servidor." });
  }
});

app.post("/api/creator-html", async (req, res) => {
  const { message, chatHistory } = req.body;
  if (!message) {
    return res.status(400).send("Por favor, proporciona un message.");
  }

  try {
    // Instantiate the model
    const model = new ChatOpenAI({
      modelName: "gpt-3.5-turbo",
      temperature: 1,
    });

    // Prompt Template
    const prompt = ChatPromptTemplate.fromMessages([
      ("system",
      "Eres un agente experto y poderoso que genera landing page con html y css"),
      new MessagesPlaceholder("chat_history"),
      ("human", "{input}"),
      new MessagesPlaceholder("agent_scratchpad"),
    ]);

    // Tools
    const retrieverWebPage = await getRetrieverFromWebPage();
    const retrieverTool = createRetrieverTool(retrieverWebPage, {
      name: "utel_search",
      description: "Usa esta herramienta para crear landing page",
    });

    const tools = [retrieverTool];
    const agent = await createOpenAIFunctionsAgent({
      llm: model,
      prompt,
      tools,
    });

    // Create the executor
    const agentExecutor = new AgentExecutor({
      agent,
      tools,
    });

    const response = await agentExecutor.invoke({
      input: `${message}.Entrega todo el contenido para agreagr en un archivo .html. 
      El archivo .html debe contener los estilos css con una etiqueta <style> dentro del mismo archivo html.
      La landing page mínimo de contener 5 secciones.
      Toma de base las herramientas que tienes para basarte en ese diseño`,
      chat_history: chatHistory,
    });
    chatHistory.push(new HumanMessage(message));
    chatHistory.push(new AIMessage(response.output));
    res.status(200).json({
      responseModel: response.output,
      chatHistory: chatHistory,
    });
  } catch (error) {
    if (error.response) {
      console.error("Error al llamar a OPENAI:", error);
      return res.status(500).json({ error: "Error en OPENAI." });
    }
    console.error("Error:", error.message);
    res.status(500).json({ error: "Error en el servidor." });
  }
});

app.listen(PORT, () => {
  console.log(`Servidor corriendo en http://localhost:${PORT}`);
});
