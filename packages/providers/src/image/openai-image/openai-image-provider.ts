import OpenAI from "openai";
import type {
  ImageProvider,
  ImageGenerationRequest,
  ImageGenerationResult,
} from "../interfaces.js";

export interface OpenAIImageProviderConfig {
  apiKey: string;
  baseUrl?: string;
}

export class OpenAIImageProvider implements ImageProvider {
  readonly name = "openai-image";
  private client: OpenAI;
  private baseUrl?: string;

  constructor(config: OpenAIImageProviderConfig) {
    this.baseUrl = config.baseUrl;
    this.client = new OpenAI({
      apiKey: config.apiKey,
      ...(config.baseUrl ? { baseURL: config.baseUrl } : {}),
    });
  }

  async generateImage(request: ImageGenerationRequest): Promise<ImageGenerationResult> {
    const model = request.model ?? "gpt-image-1";
    const size = this.resolveSize(request.width, request.height);

    const response = await this.client.images.generate({
      model,
      prompt: request.prompt,
      n: request.numImages ?? 1,
      size: size as "1024x1024" | "1024x1536" | "1536x1024" | "auto",
      quality: "high",
    });

    return {
      images: (response.data ?? []).map((img) => ({
        url: img.url ?? undefined,
        base64: img.b64_json ?? undefined,
        revisedPrompt: img.revised_prompt ?? undefined,
      })),
      model,
      provider: this.name,
    };
  }

  getSupportedModels(): string[] {
    return ["gpt-image-1"];
  }

  getDefaultSize(): { width: number; height: number } {
    return { width: 1024, height: 1536 };
  }

  private resolveSize(width?: number, height?: number): string {
    if (width && height) {
      if (width === height) return "1024x1024";
      if (width > height) return "1536x1024";
      return "1024x1536";
    }
    return "1024x1536"; // portrait default for VN
  }
}
