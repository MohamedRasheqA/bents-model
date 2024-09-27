const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const axios = require('axios');
const { pool, connectDb } = require('./config/dbConnection.cjs');
const app = express();
const port = process.env.PORT || 5002;

// Enhanced CORS configuration
const allowCors = fn => async (req, res) => {
  res.setHeader('Access-Control-Allow-Credentials', true);
  res.setHeader('Access-Control-Allow-Origin', '*');
  // Alternatively, you can use the following to restrict to specific origins:
  // res.setHeader('Access-Control-Allow-Origin', 'https://bents-model.vercel.app, https://bents-model-4ppw.vercel.app');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader(
    'Access-Control-Allow-Headers',
    'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, Authorization'
  );

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  return await fn(req, res);
};

// Middleware
app.use(bodyParser.json());
app.use(cors());

// Update FLASK_BACKEND_URL
const FLASK_BACKEND_URL = 'https://bents-model-ijmx.vercel.app';

// Debugging middleware
app.use((req, res, next) => {
  console.log(`${new Date().toISOString()} - ${req.method} ${req.url}`);
  next();
});

app.get("/", (req, res) => {
  res.send("Server is running");
});

// Wrap each route with the allowCors function
app.post('/contact', allowCors(async (req, res) => {
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
}));

app.post('/chat', allowCors(async (req, res) => {
  try {
    console.log('Forwarding chat request to Flask:', req.body);
    const response = await axios.post(`${FLASK_BACKEND_URL}/chat`, req.body);
    console.log('Received response from Flask:', response.data);
    res.json(response.data);
  } catch (error) {
    console.error('Error forwarding chat request to Flask:', error);
    res.status(500).json({ message: 'An error occurred while processing your chat request.' });
  }
}));

app.get('/documents', allowCors(async (req, res) => {
  try {
    console.log('Fetching documents from Flask');
    const response = await axios.get(`${FLASK_BACKEND_URL}/documents`);
    console.log('Received documents from Flask:', response.data);
    res.json(response.data);
  } catch (error) {
    console.error('Error fetching documents from Flask:', error);
    res.status(500).json({ message: 'An error occurred while fetching documents.' });
  }
}));

app.post('/add_document', allowCors(async (req, res) => {
  try {
    console.log('Adding document through Flask:', req.body);
    const response = await axios.post(`${FLASK_BACKEND_URL}/add_document`, req.body);
    console.log('Document added successfully:', response.data);
    res.json(response.data);
  } catch (error) {
    console.error('Error adding document through Flask:', error);
    res.status(500).json({ message: 'An error occurred while adding the document.' });
  }
}));

app.post('/delete_document', allowCors(async (req, res) => {
  try {
    console.log('Deleting document through Flask:', req.body);
    const response = await axios.post(`${FLASK_BACKEND_URL}/delete_document`, req.body);
    console.log('Document deleted successfully:', response.data);
    res.json(response.data);
  } catch (error) {
    console.error('Error deleting document through Flask:', error);
    res.status(500).json({ message: 'An error occurred while deleting the document.' });
  }
}));

app.post('/update_document', allowCors(async (req, res) => {
  try {
    console.log('Updating document through Flask:', req.body);
    const response = await axios.post(`${FLASK_BACKEND_URL}/update_document`, req.body);
    console.log('Document updated successfully:', response.data);
    res.json(response.data);
  } catch (error) {
    console.error('Error updating document through Flask:', error);
    res.status(500).json({ message: 'An error occurred while updating the document.' });
  }
}));

// Global error handler
app.use((err, req, res, next) => {
  console.error(err.stack);
  res.status(500).send('Something broke!');
});

// Start the server
app.listen(port, () => {
  console.log(`Express server is running on http://localhost:${port}`);
});

module.exports = app;
