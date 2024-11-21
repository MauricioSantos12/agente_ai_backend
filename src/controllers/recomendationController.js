

const { GoogleGenerativeAI } = require("@google/generative-ai");
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

const recomedationController = {};


recomedationController.getFinalRecomendation = async (req, res) => {
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
}

recomedationController.getFinalAsesorRecomendation = async (req, res) => {
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
}


module.exports = {
    recomedationController,
  };