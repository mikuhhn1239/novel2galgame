export interface ImageGenerationRequest {
  prompt: string;
  negativePrompt?: string;
  width?: number;
  height?: number;
  numImages?: number;
  model?: string;
  style?: string;
}

export interface GeneratedImage {
  url?: string;
  base64?: string;
  revisedPrompt?: string;
  metadata?: Record<string, unknown>;
}

export interface ImageGenerationResult {
  images: GeneratedImage[];
  model: string;
  provider: string;
  usage?: { promptTokens?: number; totalTokens?: number };
}

export type ImageGenerationStatus = "pending" | "processing" | "completed" | "failed";

export interface ImageGenerationTask {
  taskId: string;
  status: ImageGenerationStatus;
  request: ImageGenerationRequest;
  result?: ImageGenerationResult;
  error?: string;
  createdAt: string;
  completedAt?: string;
}

export interface ImageProvider {
  readonly name: string;
  generateImage(request: ImageGenerationRequest): Promise<ImageGenerationResult>;
  generateImageAsync?(request: ImageGenerationRequest): Promise<ImageGenerationTask>;
  checkTaskStatus?(taskId: string): Promise<ImageGenerationTask>;
  getSupportedModels(): string[];
  getDefaultSize(): { width: number; height: number };
}
