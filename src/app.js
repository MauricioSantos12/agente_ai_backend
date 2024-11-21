
require("dotenv").config();
const express = require('express');
const app = express();
const cors = require("cors");


app.use(express.json()); // Para parsear JSON

app.use(cors());
app.use('/', require('./routes/main.router'));

module.exports = app;
