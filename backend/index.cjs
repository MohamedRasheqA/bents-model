const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const axios = require('axios');
const { pool, connectDb } = require('./config/dbConnection.cjs');
const app = express();
const port = 5002;

const corsOptions = {
  origin: ['https://bents-model.vercel.app'],
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
};

// Middleware
app.use(bodyParser.json());

// Connect to the database
//connectDb();

// Flask backend URL
const FLASK_BACKEND_URL = 'https://bents-model-ijmx.vercel.app/';

app.get("/", (req, res) => {
  res.send("Server is running");
});

// Apply CORS to specific routes
app.post('/contact', cors(corsOptions), async (req, res) => {
  const { name, email, subject, message } = req.body;
  try {
    const query = 'INSERT INTO contacts(name, email, subject, message) VALUES($1, $2, $3, $4) RETURNING *';
    const values = [name, email, subject, message];
    const result = await pool.query(query, values);
    res.json({ message: 'Message received successfully!', data: result.rows[0] });
  } catch (err) {
    console.error('Error saving contact data:', err);
    res.status(500).json({ message: 'An error occurred while processing your request.' });
  }
});

app.post('/chat', cors(corsOptions), async (req, res) => {
  try {
    const response = await axios.post(`${FLASK_BACKEND_URL}/chat`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('Error forwarding chat request to Flask:', error);
    res.status(500).json({ message: 'An error occurred while processing your chat request.' });
  }
});

app.get('/documents', cors(corsOptions), async (req, res) => {
  try {
    const response = await axios.get(`${FLASK_BACKEND_URL}/documents`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching documents from Flask:', error);
    res.status(500).json({ message: 'An error occurred while fetching documents.' });
  }
});

app.post('/add_document', cors(corsOptions), async (req, res) => {
  try {
    const response = await axios.post(`${FLASK_BACKEND_URL}/add_document`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('Error adding document through Flask:', error);
    res.status(500).json({ message: 'An error occurred while adding the document.' });
  }
});

app.post('/delete_document', cors(corsOptions), async (req, res) => {
  try {
    const response = await axios.post(`${FLASK_BACKEND_URL}/delete_document`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('Error deleting document through Flask:', error);
    res.status(500).json({ message: 'An error occurred while deleting the document.' });
  }
});

app.post('/update_document', cors(corsOptions), async (req, res) => {
  try {
    const response = await axios.post(`${FLASK_BACKEND_URL}/update_document`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('Error updating document through Flask:', error);
    res.status(500).json({ message: 'An error occurred while updating the document.' });
  }
});

// Start the server
app.listen(port, () => {
  console.log(`Express server is running on http://localhost:${port}`);
});
