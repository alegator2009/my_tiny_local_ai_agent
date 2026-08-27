export type SessionStatus = 'active' | 'archived' | 'deleted';

export type Session = {
  id: string;
  title: string;
  description?: string | null;
  created_at: string;
  updated_at: string;
  status: SessionStatus;
  workspace_path?: string | null;
  last_window_id?: string | null;
  total_message_count: number;
  total_token_count: number;
  thinking_mode?: 'off' | 'low' | 'medium' | 'high';
  message_prefix_prompt?: string;
};

export type MessageType =
  | 'user'
  | 'assistant'
  | 'system'
  | 'tool_call'
  | 'tool_result'
  | 'internal_event';

export type Artifact = {
  id: string;
  file_name: string;
  mime_type: string;
  size_bytes: number;
  sha256: string;
  download_url: string;
  workspace_path?: string;
  workspace_abs_path?: string;
  artifact_path?: string;
  created_at?: string;
};

export type Message = {
  id: string;
  session_id: string;
  window_id: string;
  role: string;
  timestamp: string;
  content_text: string;
  content_json?: Record<string, unknown>;
  artifacts?: Artifact[];
  token_count: number;
  message_type: MessageType;
  is_pinned: boolean;
  is_anchor: boolean;
};

export type WindowState = {
  session_id: string;
  window_id: string;
  token_limit: number;
  used_tokens: number;
  used_percent: number;
  pre_rollover_threshold: number;
  hard_rollover_threshold: number;
};

export type SSEEventName =
  | 'message_delta'
  | 'model_status'
  | 'tool_status'
  | 'retrieval_status'
  | 'rollover_status'
  | 'error'
  | 'final_message';

export type ChatStreamEvent<T = unknown> = {
  event: SSEEventName;
  data: T;
};

export type RunStatus = 'queued' | 'running' | 'completed' | 'failed' | 'canceled';

export type Run = {
  id: string;
  session_id: string;
  window_id?: string | null;
  user_message_id?: string | null;
  result_message_id?: string | null;
  task_text: string;
  status: RunStatus;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  progress_json: Record<string, unknown>;
  error_text?: string | null;
};

export type RunEvent = {
  id: string;
  session_id: string;
  run_id: string;
  step_index?: number | null;
  event_type: string;
  title: string;
  detail: string;
  payload_json: Record<string, unknown>;
  timestamp: string;
};

export type RunArtifact = {
  id: string;
  session_id: string;
  run_id: string;
  artifact_id?: string | null;
  step_index?: number | null;
  stage: string;
  title: string;
  path: string;
  metadata_json: Record<string, unknown>;
  created_at: string;
};
