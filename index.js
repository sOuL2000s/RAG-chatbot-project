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
import { PGVectorStore } from '@langchain/pgvector';
import { Pool } from 'pg'; // For direct PG connection if needed

// Optional: Document Loaders (install @langchain/community for these)
// Uncomment and create a 'data' folder with 'example.pdf' if you want to use PDF loading.
// import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
// import { TextLoader } from '@langchain/community/document_loaders/fs/text';

const app = express();
const port = process.env.PORT || 3000; // Use Render's assigned port in production

// --- CORS Configuration ---
// In production, `process.env.NODE_ENV` will be 'production' on Render.
// Set 'origin' to your Netlify frontend URL. For local dev, '*' is fine.
app.use(cors({
    origin: process.env.NODE_ENV === 'production' ? 'https://your-netlify-app-domain.netlify.app' : '*',
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type'],
}));
app.use(express.json()); // For parsing application/json requests

// --- RAG Components Initialization ---
const embeddings = new GoogleGenerativeAIEmbeddings({
    model: 'embedding-001', // Specify the embedding model
    apiKey: process.env.GOOGLE_API_KEY,
});

const llm = new ChatGoogleGenerativeAI({
    model: 'gemini-pro',
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

let vectorstore; // Will hold our PGVectorStore instance

// --- Data Ingestion Function ---
// This function will add your custom knowledge to Supabase.
// It will only run if the 'documents' table is empty.
async function ingestData() {
    console.log('Starting data ingestion...');

    // A. Load Documents from various sources
    const documents = [];

    // 1. Example: Add some simple text directly
    // Add all your custom knowledge here as objects with `pageContent` and `metadata`.
    documents.push({
        pageContent: "The company's vacation policy allows 15 days of paid time off per year for full-time employees. New employees are eligible after 90 days of employment. Unused days can roll over up to a maximum of 5 days. For part-time employees, vacation days are prorated.",
        metadata: { source: 'HR Handbook', category: 'Policies', documentId: 'HR001' },
    });
    documents.push({
        pageContent: "Our Q3 earnings for 2023 showed a 10% increase in revenue compared to Q2, reaching $150 million. Profit margins remained stable at 25%. Key growth areas included cloud services and AI solutions. The CEO expressed optimism for Q4.",
        metadata: { source: 'Financial Report - Q3 2023', category: 'Finance', documentId: 'FIN003' },
    });
    documents.push({
        pageContent: "The product 'Starlight Wanderer' features a 12MP camera, 256GB storage, and a 6.7-inch OLED display. It launched on October 26, 2023, with a price of $799. Pre-orders include a free protective case.",
        metadata: { source: 'Product Specs - Starlight Wanderer', category: 'Products', product_id: 'SW001' },
    });
    documents.push({
        pageContent: "The return policy allows for returns within 30 days of purchase for a full refund, provided the item is in its original condition. Electronic items must be unopened. After 30 days, only store credit is issued for up to 60 days. Custom items are non-returnable.",
        metadata: { source: 'Customer Service - Returns', category: 'Policies', documentId: 'CS002' },
    });
    documents.push({
        pageContent: "To troubleshoot network issues, first restart your router and modem. If the problem persists, check cable connections. For advanced diagnostics, log into your router's admin panel.",
        metadata: { source: 'Technical Support Guide', category: 'Support', topic: 'Network' },
    });
    documents.push({
        pageContent: "Our mission is to innovate sustainable technology solutions that empower businesses and individuals. We are committed to ethical AI development and environmental responsibility.",
        metadata: { source: 'Company Vision', category: 'About Us' },
    });

    // 2. Example: Load from a local PDF (You'd need a 'data' folder in your backend repo with 'example.pdf')
    // If you uncomment this, make sure 'data/example.pdf' is in your rag-chatbot-backend directory
    // and committed to your Git repo for Render to access it.
    /*
    try {
        const pdfLoader = new PDFLoader('./data/example.pdf'); // Ensure example.pdf exists
        const pdfDocs = await pdfLoader.load();
        documents.push(...pdfDocs);
        console.log(`Loaded ${pdfDocs.length} documents from PDF.`);
    } catch (error) {
        console.warn("Could not load PDF from './data/example.pdf'. Ensure the file exists and is accessible.", error.message);
    }
    */

    // B. Split Documents into Chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,       // Max characters per chunk
        chunkOverlap: 200,     // Overlap between chunks to maintain context
    });
    const splitDocs = await textSplitter.splitDocuments(documents);
    console.log(`Split into ${splitDocs.length} chunks.`);

    // C. Store Embeddings in Supabase (PGVector)
    vectorstore = await PGVectorStore.fromDocuments(
        splitDocs,
        embeddings,
        {
            pool: pgPool,       // Use the shared pool
            tableName: 'documents', // Your table name in Supabase
            columns: {
                idColumnName: 'id',
                vectorColumnName: 'embedding',
                contentColumnName: 'content',
                metadataColumnName: 'metadata',
            },
        }
    );
    console.log('PGVectorStore initialized and data ingested into Supabase!');
}

// --- Define the RAG Chain ---
// This sequence defines how a user query leads to a Gemini response.
async function setupRAGChain() {
    if (!vectorstore) {
        console.error("PGVectorStore not initialized. Cannot set up RAG chain.");
        return null;
    }

    const retriever = vectorstore.asRetriever({ k: 3 }); // Retrieve top 3 most relevant document chunks

    // This prompt is crucial! It tells Gemini to ONLY use the provided context.
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
            // First, retrieve relevant documents based on the question.
            // Then, map them to extract just their content and join them.
            context: retriever.pipe(docs => docs.map(doc => doc.pageContent).join("\n\n")),
            // Pass the original question through unchanged.
            question: new RunnablePassthrough(),
        },
        // Format the retrieved context and question into our prompt template.
        promptTemplate,
        // Send the formatted prompt to the Gemini LLM.
        llm,
        // Parse Gemini's output into a simple string.
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
        const response = await ragChainInstance.invoke(query);
        console.log('Gemini raw response:', response);

        // To provide sources to the frontend, we run similaritySearch again.
        // In a more complex setup, you might enhance the ragChain to return docs directly.
        const relevantDocs = await vectorstore.similaritySearch(query, 3);
        const sources = relevantDocs.map(doc => doc.metadata?.source || 'Unknown Source');
        const uniqueSources = [...new Set(sources)]; // Get unique sources for cleaner display

        res.json({ answer: response, sources: uniqueSources });
    } catch (error) {
        console.error('Error during RAG chain invocation:', error);
        res.status(500).json({ error: 'An error occurred while processing your request.' });
    }
});

// --- Start Server and Initialize RAG Components ---
app.listen(port, async () => {
    console.log(`Server listening on port ${port}`);

    // Check if the 'documents' table in Supabase is empty.
    // This ensures data is ingested only once on the very first deploy/run.
    const client = await pgPool.connect();
    try {
        const res = await client.query('SELECT COUNT(*) FROM documents');
        if (parseInt(res.rows[0].count) === 0) {
            console.log("Supabase 'documents' table is empty. Starting data ingestion...");
            await ingestData();
        } else {
            console.log("Supabase 'documents' table already contains data. Skipping ingestion.");
            // If data exists, just initialize the vectorstore for retrieval from existing data.
            vectorstore = await PGVectorStore.fromExistingIndex(embeddings, {
                pool: pgPool,
                tableName: 'documents',
                columns: {
                    idColumnName: 'id',
                    vectorColumnName: 'embedding',
                    contentColumnName: 'content',
                    metadataColumnName: 'metadata',
                },
            });
            console.log('PGVectorStore initialized from existing Supabase index!');
        }
    } catch (error) {
        console.error('Error during initial Supabase check or vectorstore initialization:', error);
        // If there's an error (e.g., table doesn't exist on first run before ingestData creates it),
        // we can attempt ingestion anyway, or simply initialize for retrieval if data exists.
        // For robustness in this setup, if any error, we'll try to re-ingest (which will add if not present).
        console.warn("Attempting data ingestion due to an error or empty table condition...");
        await ingestData(); // This will ensure the table and data are set up if not already.
    } finally {
        client.release(); // Release the client back to the pool
    }

    ragChainInstance = await setupRAGChain();
    if (ragChainInstance) {
        console.log('RAG chain is ready to answer questions!');
    }
});