// Vercel serverless entrypoint.
// Rebuilds the same Express app (see server.js) as a single serverless
// function. Storage, RAG, and workspace writes silently no-op because the
// serverless filesystem is read-only — this is a demo shell deployment.
// Full capabilities require a persistent server (local PC or Render).
const { app } = require('../server');

module.exports = app;