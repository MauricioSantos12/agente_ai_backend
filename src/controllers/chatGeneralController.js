const {
  TavilySearchResults,
} = require("@langchain/community/tools/tavily_search");
const { ChatOpenAI } = require("@langchain/openai");
const {
  ChatPromptTemplate,
  MessagesPlaceholder,
} = require("@langchain/core/prompts");

const {
  createOpenAIFunctionsAgent,
  AgentExecutor,
} = require("langchain/agents");

const { HumanMessage, AIMessage } = require("@langchain/core/messages");
const chatGeneralController = {};

chatGeneralController.generalChat = async (req, res) => {
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
};

module.exports = {
  chatGeneralController,
};
