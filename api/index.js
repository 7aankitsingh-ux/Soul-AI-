// Vercel serverless entrypoint.
// Rebuilds the same Express app (see server.js) as a single serverless
// function. Storage, RAG, and workspace writes silently no-op because the
// serverless filesystem is read-only — this is a demo shell deployment.
// Full capabilities require a persistent server (local PC or Render).
const { app } = require('../server');

// Future-proofing: if Vercel ever hands this function the bare rewritten
// destination path (e.g. "/api") instead of the original request URL, recover
// the browser-visible path from the rewrite headers and let Express route on
// it. When the headers are absent (current legacy behavior) this is a no-op.
app.use((req, res, next) => {
  const original = req.headers['x-vercel-rewrite-url'] ||
                   req.headers['x-rewrite-url'] ||
                   req.headers['x-original-url'];
  if (original && original !== req.url) {
    req.url = req.originalUrl = original;
  }
  next();
});

module.exports = app;