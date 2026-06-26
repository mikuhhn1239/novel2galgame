export interface VideoGenerationRequest {
  prompt: string;
  negativePrompt?: string;
  width?: number;
  height?: number;
  numFrames?: number;
  frameRate?: number;
  model?: string;
  /** Single image URL for image-to-video */
  imageUrl?: string;
  /** Multiple image URLs for multi-image or keyframe animation */
  imageUrls?: string[];
  /** Generation mode, e.g. "keyframes" */
  mode?: string;
  seed?: number;
  inferenceSteps?: number;
}

export type VideoGenerationStatus = "queued" | "in_progress" | "completed" | "failed";

export interface VideoGenerationTask {
  taskId: string;
  videoId: string;
  status: VideoGenerationStatus;
  progress: number;
  request: VideoGenerationRequest;
  videoUrl?: string;
  error?: string;
  seconds?: string;
  size?: string;
  createdAt: string;
  completedAt?: string;
}

export interface VideoProvider {
  readonly name: string;
  /** Create an async video generation task. Returns task info immediately. */
  createTask(request: VideoGenerationRequest): Promise<VideoGenerationTask>;
  /** Poll task status. Returns updated task. */
  checkTaskStatus(taskId: string): Promise<VideoGenerationTask>;
  getSupportedModels(): string[];
  getDefaultSize(): { width: number; height: number };
}
