const { createOpenAIFunctionsAgent, AgentExecutor } = require("langchain/agents");
const { ChatOpenAI } = require("langchain/chat_models/openai");
const { ChatPromptTemplate } = require("langchain/prompts");
const { HumanMessage, AIMessage } = require("langchain/schema");
const { createRetrieverTool } = require("langchain/tools/retriever");

const creatorController = {};

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

creatorController.creatorHtml = async (req, res) => {
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
}



module.exports = {
    creatorController,
  };