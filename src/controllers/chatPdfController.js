const { ChatOpenAI } = require("@langchain/openai");
const fs = require("fs");
const pdf = require("pdf-parse");
const { Document } = require("langchain/document");
const {
  ChatPromptTemplate,
  MessagesPlaceholder,
} = require("@langchain/core/prompts");
const {
    TavilySearchResults,
  } = require("@langchain/community/tools/tavily_search");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { OpenAIEmbeddings } = require("@langchain/openai");
const { HumanMessage, AIMessage } = require("@langchain/core/messages");
const { MemoryVectorStore } = require("langchain/vectorstores/memory");
const chatPDFGeneralController = {};
const { createRetrieverTool } = require("langchain/tools/retriever");
const {
    createOpenAIFunctionsAgent,
    AgentExecutor,
  } = require("langchain/agents");
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

chatPDFGeneralController.chatPdf = async (req, res) => {
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
};

module.exports = {
  chatPDFGeneralController,
};
