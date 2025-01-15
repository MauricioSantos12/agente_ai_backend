const express = require("express");
const cors = require("cors");
const fs = require("fs");
const pdf = require("pdf-parse");
const { GoogleGenerativeAI } = require("@google/generative-ai");
const { ChatOpenAI } = require("@langchain/openai");
const {
  ChatPromptTemplate,
  MessagesPlaceholder,
} = require("@langchain/core/prompts");
const {
  HumanMessage,
  AIMessage,
  SystemMessage,
} = require("@langchain/core/messages");
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
const PORT = process.env.PORT || 8080;

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

const getRetrieverFromPDFs = async (textPdf) => {
  // Cargar el contenido de los PDFs
  const pdfData = await loadPdfDataUtel();
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 700,
    chunkOverlap: 100,
  });

  // Dividir cada documento en fragmentos
  const splitDocs = [];

  if (textPdf) {
    const doc = new Document({
      pageContent: textPdf,
      metadata: { name: "pdf-by-lp" },
    });
    const chunks = await splitter.splitDocuments([doc]);
    splitDocs.push(...chunks);
  } else {
    for (const [name, content] of Object.entries(pdfData)) {
      const doc = new Document({
        pageContent: content,
        metadata: { name },
      });
      const chunks = await splitter.splitDocuments([doc]);
      splitDocs.push(...chunks);
    }
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

const getHtmlByTypeLp = (typeLp) => {
  let filePath = "";
  switch (typeLp ? typeLp.toLowerCase().trim() : "") {
    case "femsa":
      filePath = "./base_lps/femsa/index.html";
      break;
    case "ebook":
      filePath = "./base_lps/ebook/index.html";
      break;
    case "webinar":
      filePath = "./base_lps/webinar/index.html";
      break;
    case "webinar_ingenieria":
      filePath = "./base_lps/webinar_ingenieria/index.html";
      break;
    default:
      "./base_lps/shcp/index.html";
      break;
  }
  return new Promise((resolve, reject) => {
    fs.readFile(filePath, "utf-8", (error, data) => {
      if (error) {
        console.error("Error al cargar el archivo:", error);
        reject(error);
      } else {
        resolve(data);
      }
    });
  });
};

const getStylesByTypeLp = (typeLp) => {
  let filePath = "";
  switch (typeLp ? typeLp.toLowerCase().trim() : "") {
    case "femsa":
      filePath = "./base_lps/femsa/styles.css";
      break;
    case "ebook":
      filePath = "./base_lps/ebook/styles.css";
      break;
    case "webinar":
      filePath = "./base_lps/webinar/styles.css";
      break;
    case "webinar_ingenieria":
      filePath = "./base_lps/webinar_ingenieria/styles.css";
      break;
    default:
      "./base_lps/shcp/styles.css";
      break;
  }

  return new Promise((resolve, reject) => {
    fs.readFile(filePath, "utf-8", (error, data) => {
      if (error) {
        console.error("Error al cargar el archivo:", error);
        reject(error);
      } else {
        resolve(data);
      }
    });
  });
};

const mergeHtmlAndCss = (html, css) => {
  const htmlSplitted = html.split("</head>");
  let finalHtml;
  if (htmlSplitted.length > 0) {
    finalHtml = `
    ${htmlSplitted[0]}
    <style>
    ${css}
    </style>
    ${htmlSplitted[1]}
  `;
  }
  return finalHtml;
};

app.get("/", (req, res) => res.send("Express on Render"));

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
  const { message, chatHistory, typeLp, utm } = req.body;
  const withPdf = false;

  if (!message) {
    return res.status(400).send("Por favor, proporciona un message.");
  }

  const model = new ChatOpenAI({
    temperature: 1,
    modelName: "gpt-3.5-turbo",
  });

  try {
    const finalResponse = {
      output: "",
      chatHistory: chatHistory,
    };
    let agentExecutor;
    let response;
    let topic;
    let finalTypeLp;
    if (utm) {
      const url = new URL(message);
      topic = url.searchParams.get("utm_topic");
      finalTypeLp = url.searchParams.get("utm_typeLp");
    } else {
      topic = message;
      finalTypeLp = typeLp;
    }
    const htmlContent = await getHtmlByTypeLp(finalTypeLp);
    const cssContent = await getStylesByTypeLp(finalTypeLp);
    if (!htmlContent || htmlContent.trim() === "") {
      throw new Error(
        "El archivo HTML está vacío o no se cargó correctamente."
      );
    }

    if (withPdf) {
      const prompt = ChatPromptTemplate.fromMessages([
        new SystemMessage(
          `Eres un experto en generación de landing pages. 
          Tu tarea es modificar TODOS los textos visibles en el archivo HTML proporcionado para que estén relacionados con el tema "${topic}". 
          NO debes cambiar la estructura del archivo HTML, los nombres de las clases, los IDs ni ningún estilo.

          ### Cambios que debes realizar:
          1. Actualiza los títulos (<h1>, <h2>, <h3>, etc.) para que estén alineados con el tema "${topic}".
          2. Modifica los párrafos (<p>) para que reflejen contenido relevante al tema.
          3. Ajusta los textos de botones (<a>, <button>) para que coincidan con el tema.
          4. Cambia los atributos "alt" y "title" de imágenes para describirlas de acuerdo al tema.

          ### Requisitos:
          - Devuelve SOLO el archivo HTML completo, sin explicaciones ni comentarios adicionales. 
          - Asegúrate de que el archivo HTML sea válido y comience con "<!DOCTYPE html>".
          - No omitas ningún contenido ni estilo existente en el HTML base.
          - Asegúrate de que el texto modificado sea coherente, persuasivo y atractivo en el contexto del tema proporcionado.

          ### Archivo HTML Base:
          ${htmlContent}`
        ),
        new MessagesPlaceholder("chat_history"),
        new MessagesPlaceholder("agent_scratchpad"),
      ]);
      const retrieverUtelPdf = await getRetrieverFromPDFs();
      const retrieverTool = createRetrieverTool(retrieverUtelPdf, {
        name: "utel_search",
        description: "Usa esta herramienta para buscar información",
      });

      const agent = await createOpenAIFunctionsAgent({
        llm: model,
        prompt,
        tools: [retrieverTool],
      });

      agentExecutor = new AgentExecutor({
        agent,
        tools: [retrieverTool],
      });

      response = await agentExecutor.invoke({
        input: message,
        chat_history: chatHistory || [],
      });
      if (response) {
        finalResponse.output = response.output;
      }
    } else {
      const prompt = ChatPromptTemplate.fromMessages([
        new SystemMessage(
          `
          Eres un experto en generación de landing pages. 
          Tu tarea es modificar TODOS los textos visibles en el archivo HTML proporcionado para que estén relacionados con el tema "${topic}". 
          NO debes cambiar la estructura del archivo HTML, los nombres de las clases, los IDs ni ningún estilo.

          ### Cambios que debes realizar:
          1. Actualiza los títulos (<h1>, <h2>, <h3>, etc.) para que estén alineados con el tema "${topic}".
          2. Modifica los párrafos (<p>) para que reflejen contenido relevante al tema.
          3. Ajusta los textos de botones (<a>, <button>) para que coincidan con el tema.
          4. Cambia los atributos "alt" y "title" de imágenes para describirlas de acuerdo al tema.

          ### Requisitos:
          - Devuelve SOLO el archivo HTML completo, sin explicaciones ni comentarios adicionales. 
          - Asegúrate de que el archivo HTML sea válido y comience con "<!DOCTYPE html>".
          - No omitas ningún contenido ni estilo existente en el HTML base.
          - Asegúrate de que el texto modificado sea coherente, persuasivo y atractivo en el contexto del tema proporcionado.

          ### Archivo HTML Base:
          ${htmlContent}
          `
        ),
        new MessagesPlaceholder("chat_history"),
        new HumanMessage("{htmlData}"),
      ]);
      const chain = prompt.pipe(model);

      response = await chain.invoke({
        input: message,
        chat_history: chatHistory,
      });
      if (response) {
        finalResponse.output = response.content;
      }
    }

    if (!finalResponse || !finalResponse.output) {
      console.error("No se recibió respuesta de la IA.");
      return res.status(500).json({ error: "Respuesta vacía de la IA." });
    }

    if (!finalResponse.output.startsWith("<!DOCTYPE html>")) {
      console.error("La IA no devolvió HTML válido:");
      return res.status(500).json({ error: "HTML no válido." });
    }
    const htmlToGenerate = mergeHtmlAndCss(finalResponse.output, cssContent);

    res.status(200).json({
      htmlToRender: htmlToGenerate,
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
  console.log(`Servidor en http://localhost:${PORT}`);
});
