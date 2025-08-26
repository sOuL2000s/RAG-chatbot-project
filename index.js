// rag-chatbot-backend/index.js

import 'dotenv/config'; // Load environment variables from .env file
import express from 'express';
import cors from 'cors';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { PromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
import { RunnablePassthrough, RunnableSequence } from '@langchain/core/runnables';
import { Document } from '@langchain/core/documents'; // To work with LangChain Document objects
import { Pool } from 'pg'; // For direct PG connection

// Optional: Document Loaders (install @langchain/community for these)
// Uncomment and create a 'data' folder with 'example.pdf' if you want to use PDF loading.
// import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';

const app = express();
const port = process.env.PORT || 3000; // Use Render's assigned port in production

// --- CORS Configuration ---
app.use(cors({
    origin: process.env.NODE_ENV === 'production' ? 'https://your-netlify-app-domain.netlify.app' : '*',
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type'],
}));
app.use(express.json());

// --- RAG Components Initialization ---
const embeddings = new GoogleGenerativeAIEmbeddings({
    model: 'embedding-001', // Specify the embedding model
    apiKey: process.env.GOOGLE_API_KEY,
});

const llm = new ChatGoogleGenerativeAI({
    model: 'gemini-1.5-flash-preview-0520', // Using the specified Flash model
    temperature: 0.7, // Adjust for creativity vs. consistency (0.0 - 1.0)
    apiKey: process.env.GOOGLE_API_KEY,
});

// --- Supabase PGVector Configuration ---
const pgConfig = {
    connectionString: process.env.PG_CONNECTION_STRING,
    ssl: {
        rejectUnauthorized: false // Often required for Render -> Supabase connections
    }
};
const pgPool = new Pool(pgConfig); // Create a PostgreSQL connection pool once

// --- Custom Retriever Function ---
// This replaces the LangChain PGVectorStore.asRetriever()
async function retrieveDocuments(query) {
    const queryEmbedding = await embeddings.embedQuery(query);

    const client = await pgPool.connect();
    try {
        const res = await client.query(
            `SELECT
                id,
                content,
                metadata,
                embedding <=> $1 AS distance
             FROM documents
             ORDER BY distance
             LIMIT 3;`, // Retrieve top 3 most relevant documents
            [JSON.stringify(queryEmbedding)] // pg_vector expects JSON string for vector input
        );

        // Convert query results back into LangChain Document objects
        return res.rows.map(row => new Document({
            pageContent: row.content,
            metadata: row.metadata,
        }));
    } finally {
        client.release();
    }
}

// --- Data Ingestion Function ---
// This function will add your custom knowledge to Supabase.
async function ingestData() {
    console.log('Starting data ingestion...');

    const documents = [];

    // 1. Example: Add some simple text directly
    documents.push(new Document({
        pageContent: "The company's vacation policy allows 15 days of paid time off per year for full-time employees. New employees are eligible after 90 days of employment. Unused days can roll over up to a maximum of 5 days. For part-time employees, vacation days are prorated.",
        metadata: { source: 'HR Handbook', category: 'Policies', documentId: 'HR001' },
    }));
    documents.push(new Document({
        pageContent: "Our Q3 earnings for 2023 showed a 10% increase in revenue compared to Q2, reaching $150 million. Profit margins remained stable at 25%. Key growth areas included cloud services and AI solutions. The CEO expressed optimism for Q4.",
        metadata: { source: 'Financial Report - Q3 2023', category: 'Finance', documentId: 'FIN003' },
    }));
    documents.push(new Document({
        pageContent: "The product 'Starlight Wanderer' features a 12MP camera, 256GB storage, and a 6.7-inch OLED display. It launched on October 26, 2023, with a price of $799. Pre-orders include a free protective case.",
        metadata: { source: 'Product Specs - Starlight Wanderer', category: 'Products', product_id: 'SW001' },
    }));
    documents.push(new Document({
        pageContent: "The return policy allows for returns within 30 days of purchase for a full refund, provided the item is in its original condition. Electronic items must be unopened. After 30 days, only store credit is issued for up to 60 days. Custom items are non-returnable.",
        metadata: { source: 'Customer Service - Returns', category: 'Policies', documentId: 'CS002' },
    }));
    documents.push(new Document({
        pageContent: "To troubleshoot network issues, first restart your router and modem. If the problem persists, check cable connections. For advanced diagnostics, log into your router's admin panel.",
        metadata: { source: 'Technical Support Guide', category: 'Support', topic: 'Network' },
    }));
    documents.push(new Document({
        pageContent: "Our mission is to innovate sustainable technology solutions that empower businesses and individuals. We are committed to ethical AI development and environmental responsibility.",
        metadata: { source: 'Company Vision', category: 'About Us' },
    }));

    // 2. Example: Load from a local PDF
    /*
    try {
        // Ensure you have a 'data' folder in your backend repo with 'example.pdf'
        // and that 'pdf-parse' is correctly used by the PDFLoader.
        const pdfLoader = new PDFLoader('./data/example.pdf');
        const pdfDocs = await pdfLoader.load();
        documents.push(...pdfDocs);
        console.log(`Loaded ${pdfDocs.length} documents from PDF.`);
    } catch (error) {
        console.warn("Could not load PDF from './data/example.pdf'. Ensure the file exists and is accessible.", error.message);
    }
    */

    // B. Split Documents into Chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
    });
    const splitDocs = await textSplitter.splitDocuments(documents);
    console.log(`Split into ${splitDocs.length} chunks.`);

    // C. Embed and Store in Supabase (PGVector)
    const client = await pgPool.connect();
    try {
        for (const doc of splitDocs) {
            const embedding = await embeddings.embedDocuments([doc.pageContent]);
            await client.query(
                `INSERT INTO documents (content, metadata, embedding) VALUES ($1, $2, $3);`,
                [doc.pageContent, doc.metadata, JSON.stringify(embedding[0])] // Store the first (only) embedding
            );
        }
        console.log(`Ingested ${splitDocs.length} document chunks into Supabase!`);
    } finally {
        client.release();
    }
}

// --- Define the RAG Chain ---
async function setupRAGChain() {
    const promptTemplate = PromptTemplate.fromTemplate(`
You are an AI assistant that answers questions based ONLY on the provided context.
If you cannot find the answer in the provided context, politely state that you don't have enough information.
Always try to be concise and helpful. When possible, mention the source of the information.

Context:
{context}

Question:
{question}

Answer:
`);

    // LangChain Expression Language (LCEL) to define the RAG flow
    const ragChain = RunnableSequence.from([
        {
            // The context is now generated by our custom retrieveDocuments function
            context: (input) => retrieveDocuments(input.question).then(docs => docs.map(doc => doc.pageContent).join("\n\n")),
            // Pass the original question through unchanged
            question: new RunnablePassthrough(),
        },
        promptTemplate,
        llm,
        new StringOutputParser(),
    ]);

    return ragChain;
}

// --- API Endpoint for Chat ---
let ragChainInstance; // Store the initialized RAG chain for reuse

app.post('/api/chat', async (req, res) => {
    const { query } = req.body;
    if (!query) {
        return res.status(400).json({ error: 'Query is required.' });
    }

    if (!ragChainInstance) {
        return res.status(500).json({ error: 'RAG chain not initialized. Please try again in a moment.' });
    }

    try {
        console.log(`Received query: "${query}"`);
        const response = await ragChainInstance.invoke({ question: query }); // Pass query as an object { question: query }
        console.log('Gemini raw response:', response);

        // Retrieve sources manually for display
        const relevantDocs = await retrieveDocuments(query);
        const sources = relevantDocs.map(doc => doc.metadata?.source || 'Unknown Source');
        const uniqueSources = [...new Set(sources)];

        res.json({ answer: response, sources: uniqueSources });
    } catch (error) {
        console.error('Error during RAG chain invocation:', error);
        res.status(500).json({ error: 'An error occurred while processing your request.' });
    }
});

// --- Start Server and Initialize RAG Components ---
app.listen(port, async () => {
    console.log(`Server listening on port ${port}`);

    const client = await pgPool.connect();
    try {
        const res = await client.query('SELECT COUNT(*) FROM documents');
        if (parseInt(res.rows[0].count) === 0) {
            console.log("Supabase 'documents' table is empty. Starting data ingestion...");
            await ingestData();
        } else {
            console.log("Supabase 'documents' table already contains data. Skipping ingestion.");
        }
    } catch (error) {
        console.error('Error during initial Supabase check. Attempting data ingestion anyway:', error);
        // This might happen if the table doesn't exist yet, so try to ingest.
        await ingestData();
    } finally {
        client.release();
    }

    ragChainInstance = await setupRAGChain();
    if (ragChainInstance) {
        console.log('RAG chain is ready to answer questions!');
    }
});