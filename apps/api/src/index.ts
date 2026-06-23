import "dotenv/config";
import dns from "node:dns";
import { createDatabase } from "@novel2gal/storage";
import { FetchLLMProvider } from "@novel2gal/providers";
import { createServer } from "./server/server.js";
import { config } from "./config/index.js";

// Force IPv4 to avoid UND_ERR_SOCKET on Windows with IPv6
dns.setDefaultResultOrder("ipv4first");

const db = createDatabase(config.dataDir);

let provider: FetchLLMProvider | null = null;
const apiKey = process.env.OPENAI_API_KEY;
if (apiKey) {
  provider = new FetchLLMProvider({
    apiKey,
    baseUrl: process.env.OPENAI_BASE_URL || "https://api.openai.com/v1",
    defaultModel: process.env.DEFAULT_MODEL ?? "gpt-4o",
    name: process.env.LLM_PROVIDER_NAME ?? "nvidia",
  });
  console.log(`LLM provider: ${provider.name} (${process.env.DEFAULT_MODEL ?? "gpt-4o"})`);
} else {
  console.log("WARNING: No OPENAI_API_KEY set. Chapter processing will be unavailable.");
}

const app = createServer(db, provider);

app.listen(config.port, () => {
  console.log(`API server running on http://localhost:${config.port}`);
  console.log(`Data directory: ${config.dataDir}`);
});
