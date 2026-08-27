// Redirect noisy stdout logs to stderr so MCP stdio channel stays protocol-clean.
console.log = (...args) => {
  console.error(...args);
};

await import("./dist/index.js");
