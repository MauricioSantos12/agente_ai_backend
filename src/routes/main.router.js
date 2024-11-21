const { chatGeneralController } = require("../controllers/chatGeneralController");
const { chatPDFGeneralController } = require("../controllers/chatPdfController");
const { questionsController } = require("../controllers/questionsController");
const { recomedationController } = require("../controllers/recomendationController");
const { asesorController } = require("../controllers/asesorController");


const express = require('express');
const router = express.Router(); // Crea una instancia del Router


// Define tus rutas aquí
router.get('/', (req, res) => {
  res.send('The server is ready');
});

// Rutas de API
router.post('/api/general-chat', chatGeneralController.generalChat);
router.post('/api/general-chat-pdf', chatPDFGeneralController.chatPdf);
router.post('/api/get-questions', questionsController.getQuestions);

router.post('/api/get-final-recomedation', recomedationController.getFinalRecomendation);
router.post('/api/get-final-asesor-recomedation', recomedationController.getFinalAsesorRecomendation);
router.post('/api/asesor', asesorController.asesor);

module.exports = router;