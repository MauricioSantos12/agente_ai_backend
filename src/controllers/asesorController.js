const { ChatOpenAI } = require("@langchain/openai");
const {
    ChatPromptTemplate,
    MessagesPlaceholder,
  } = require("@langchain/core/prompts");
  const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
  const { MemoryVectorStore } = require("langchain/vectorstores/memory");
  const { OpenAIEmbeddings } = require("@langchain/openai");  
  const { createRetrieverTool } = require("langchain/tools/retriever");
  const {
    createOpenAIFunctionsAgent,
    AgentExecutor,
  } = require("langchain/agents");

  const { HumanMessage, AIMessage } = require("@langchain/core/messages");
  const fs = require("fs");
const asesorController = {};
const pdf = require("pdf-parse");
const { Document } = require("langchain/document");
const extractPDFContent = async (filePath) => {
  const dataBuffer = fs.readFileSync(filePath);
  const data = await pdf(dataBuffer);
  return data.text;
};


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
      chunkSize: 1000,
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
      console.log(`doc`,doc);
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

asesorController.asesor = async (req, res) => {
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
}

module.exports = {
    asesorController,
  };