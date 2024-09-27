const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const axios = require('axios');
const { pool, connectDb } = require('./config/dbConnection.cjs');
const app = express();
const port = 5002;

const corsOptions = {
  origin: ['https://bents-model.vercel.app', 'https://bents-model-4ppw.vercel.app'],
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
  credentials: true,
};

app.use(cors(corsOptions));

// Update FLASK_BACKEND_URL
const FLASK_BACKEND_URL = 'https://bents-model-ijmx.vercel.app';  // Make sure this matches your actual Flask backend URL

// Add OPTIONS handling for preflight requests
app.options('*', cors(corsOptions));

// Middleware
app.use(bodyParser.json());


// Connect to the database
//connectDb();

// Flask backend URL

app.get("/",(req,res)=>
{
  res.send("Server is running");
})
// Route to handle contact form submission
app.post('/contact', async (req, res) => {
  const { name, email, subject, message } = req.body;

  try {
    // Insert form data into the PostgreSQL table
    const query = 'INSERT INTO contacts(name, email, subject, message) VALUES($1, $2, $3, $4) RETURNING *';
    const values = [name, email, subject, message];
    const result = await pool.query(query, values);

    res.json({ message: 'Message received successfully!', data: result.rows[0] });
  } catch (err) {
    console.error('Error saving contact data:', err);
    res.status(500).json({ message: 'An error occurred while processing your request.' });
  }
});

// Route to handle chat requests
app.post('/chat', async (req, res) => {
  try {
    const response = await axios.post(`${FLASK_BACKEND_URL}/chat`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('Error forwarding chat request to Flask:', error);
    res.status(500).json({ message: 'An error occurred while processing your chat request.' });
  }
});


// Route to get all documents (products)
app.get('/documents', async (req, res) => {
  try {
    const response = await axios.get(`${FLASK_BACKEND_URL}/documents`);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching documents from Flask:', error);
    res.status(500).json({ message: 'An error occurred while fetching documents.' });
  }
});

// Route to add a document (product)
app.post('/add_document', async (req, res) => {
  try {
    const response = await axios.post(`${FLASK_BACKEND_URL}/add_document`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('Error adding document through Flask:', error);
    res.status(500).json({ message: 'An error occurred while adding the document.' });
  }
});

// Route to delete a document (product)
app.post('/delete_document', async (req, res) => {
  try {
    const response = await axios.post(`${FLASK_BACKEND_URL}/delete_document`, req.body);
    res.json(response.data);
  } catch (error) {
    console.error('Error deleting document through Flask:', error);
    res.status(500).json({ message: 'An error occurred while deleting the document.' });
  }
});

// Route to update a document (product)
app.post('/update_document', async (req, res) => {
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
