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
const questionsController = {};

console.log('testxxxx');

questionsController.getQuestions = async (req, res) => {
  const { country, ageRange, areas } = req.body;
  console.log(`req.body`,req.body);

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
};



module.exports = {
    questionsController,
  };
  