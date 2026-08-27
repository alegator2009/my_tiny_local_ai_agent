export type Session = {
  id: string;
  title: string;
  description?: string | null;
  created_at: string;
  updated_at: string;
  status: string;
  workspace_path?: string | null;
  last_window_id?: string | null;
  total_message_count: number;
  total_token_count: number;
  thinking_mode: ThinkingMode;
  message_prefix_prompt: string;
  provider_id?: string | null;
  model_id?: string | null;
  // Chat-header checkboxes — persisted so the chat restores the user's
  // preferred toggle state when they switch sessions.
  hide_system_messages?: boolean;
  run_in_background?: boolean;
  force_search_next?: boolean;
  bypass_search_cache_next?: boolean;
  context_mode?: 'full' | 'skill_state';
};

export type ModelEntry = {
  id: string;
  name: string;
  display_name: string;
  context_window_size: number;
  max_output_tokens: number;
  temperature: number;
  top_p: number;
  extra_params_json: Record<string, any>;
  is_default: boolean;
  enabled: boolean;
  created_at?: string;
  updated_at?: string;
};

export type Provider = {
  id: string;
  name: string;
  provider_name: string;
  base_url: string;
  endpoint: string;
  api_key: string;
  request_timeout_sec: number;
  enabled: boolean;
  notes: string;
  models: ModelEntry[];
  created_at?: string;
  updated_at?: string;
};

export type ProviderListResponse = {
  providers: Provider[];
  active_provider_id: string | null;
  active_model_id: string | null;
};

export type MessagePrefixTemplate = {
  id: string;
  name: string;
  prompt: string;
  created_at: string;
  updated_at: string;
};

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
  role: string;
  content_text: string;
  timestamp: string;
  message_type: string;
  is_pinned: boolean;
  is_anchor: boolean;
  artifacts?: Artifact[];
  content_json?: Record<string, any>;
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

export type StreamEvent = {
  event: string;
  data: any;
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
  progress_json: Record<string, any>;
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
  payload_json: Record<string, any>;
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
  metadata_json: Record<string, any>;
  created_at: string;
};

export type EvolutionStatus = 'queued' | 'running' | 'completed' | 'failed' | 'canceled';
export type EvolutionRun = {
  id: string;
  prompt: string;
  status: EvolutionStatus;
  mode: string;
  max_generations: number;
  stop_on_failure: boolean;
  current_generation: number;
  parent_generation?: number | null;
  child_generation?: number | null;
  lineage_root_path: string;
  parent_repo_path: string;
  child_repo_path?: string | null;
  created_at: string;
  updated_at: string;
  started_at?: string | null;
  finished_at?: string | null;
  progress_json: Record<string, any>;
  score_json: Record<string, any>;
  error_text?: string | null;
};

export type EvolutionEvent = {
  id: string;
  run_id: string;
  generation?: number | null;
  event_type: string;
  title: string;
  detail: string;
  payload_json: Record<string, any>;
  timestamp: string;
};

export type EvolutionGeneration = {
  generation: number;
  name: string;
  status: 'passed' | 'failed' | 'unknown';
  active: boolean;
  artifact_dir: string;
  child_repo_path?: string | null;
  prompt: string;
  improvement_summary: string;
  mode?: string | null;
  created_at?: string | null;
  self_test_ok?: boolean | null;
  tests_ok?: boolean | null;
  has_handoff: boolean;
  deletable: boolean;
};

export type ThinkingMode = 'off' | 'low' | 'medium' | 'high';
export type ThinkingModeTurn = 'session' | ThinkingMode;
export type MCPDiscovery = {
  ok: boolean;
  tools: Array<{
    name: string;
    server_name: string;
    mcp_tool_name: string;
    description: string;
    input_schema: Record<string, any>;
  }>;
  errors: Array<{ server: string; error: string }>;
};

export type AutoSearchConfig = {
  enabled: boolean;
  policy: 'off' | 'auto' | 'always';
  max_chars: number;
  cache_ttl_sec: number;
  max_per_turn: number;
  max_citations: number;
  summary_max_chars: number;
  snippet_per_source_chars: number;
  include_snippets: boolean;
  include_full_content: boolean;
  prefer_engine: string;
  freshness_hints: string[];
  factual_hints: string[];
  opinion_hints: string[];
};

export type AutoSearchCitation = {
  title: string;
  url: string;
  description?: string;
  snippet?: string;
  engine?: string;
};

export type AutoSearchDecision = {
  should_search: boolean;
  reason: string;
  policy: string;
  query: string;
  normalized_query: string;
};

export type AutoSearchResult = {
  query: string;
  normalized_query: string;
  answer: string;
  citations: AutoSearchCitation[];
  engine: string;
  source: string;
  cache_hit: boolean;
  took_ms: number;
  error: string;
  grounded_block: string;
};

export type AutoSearchTestResponse = {
  decision: AutoSearchDecision;
  result: AutoSearchResult;
};

export function testAutoSearch(payload: { query: string; force?: boolean; bypass_cache?: boolean }) {
  return jsonFetch<AutoSearchTestResponse>('/api/settings/auto-search/test', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export function buildApiUrl(path: string): string {
  if (!path) {
    return API_URL;
  }
  return path.startsWith('http://') || path.startsWith('https://') ? path : `${API_URL}${path}`;
}

async function jsonFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers || {})
    },
    cache: 'no-store'
  });

  if (!res.ok) {
    throw new Error(`Request failed: ${res.status}`);
  }
  return (await res.json()) as T;
}

export function listSessions() {
  return jsonFetch<Session[]>('/api/sessions');
}

export function createSession(input: {
  title: string;
  description?: string;
  workspace_path?: string;
  // All of these were silently dropped before — backend `SessionCreate`
  // accepts them and stores them, so the UI should pass them through.
  thinking_mode?: ThinkingMode;
  force_search_next?: boolean;
  bypass_search_cache_next?: boolean;
  context_mode?: 'full' | 'skill_state';
  provider_id?: string | null;
  model_id?: string | null;
}) {
  return jsonFetch<Session>('/api/sessions', {
    method: 'POST',
    body: JSON.stringify(input)
  });
}

export function updateSession(
  sessionId: string,
  input: Partial<{
    title: string;
    description: string;
    workspace_path: string;
    thinking_mode: ThinkingMode;
    message_prefix_prompt: string;
    provider_id: string | null;
    model_id: string | null;
    hide_system_messages: boolean;
    run_in_background: boolean;
    force_search_next: boolean;
    bypass_search_cache_next: boolean;
    context_mode: 'full' | 'skill_state';
  }>
) {
  return jsonFetch<Session>(`/api/sessions/${sessionId}`, {
    method: 'PATCH',
    body: JSON.stringify(input)
  });
}

export function removeSession(sessionId: string) {
  return jsonFetch<{ ok: boolean }>(`/api/sessions/${sessionId}`, {
    method: 'DELETE'
  });
}

export function archiveSession(sessionId: string) {
  return jsonFetch<Session>(`/api/sessions/${sessionId}/archive`, {
    method: 'POST'
  });
}

export function listMessagePrefixTemplates() {
  return jsonFetch<MessagePrefixTemplate[]>('/api/message-prefix-templates');
}

export function saveMessagePrefixTemplate(input: { name: string; prompt: string }) {
  return jsonFetch<MessagePrefixTemplate>('/api/message-prefix-templates', {
    method: 'POST',
    body: JSON.stringify(input)
  });
}

export function removeMessagePrefixTemplate(templateId: string) {
  return jsonFetch<{ ok: boolean }>(`/api/message-prefix-templates/${templateId}`, {
    method: 'DELETE'
  });
}

export function getTranscript(sessionId: string) {
  return jsonFetch<Message[]>(`/api/chat/${sessionId}/transcript`);
}

import type { SessionGraph } from './graphTypes';

export function getSessionGraph(sessionId: string) {
  return jsonFetch<SessionGraph>(`/api/sessions/${sessionId}/graph`);
}

export function getWindowState(sessionId: string) {
  return jsonFetch<WindowState>(`/api/chat/${sessionId}/window-state`);
}

export function getSettings() {
  return jsonFetch<any>('/api/settings');
}

export function updateSettings(payload: any) {
  return jsonFetch<any>('/api/settings', {
    method: 'PUT',
    body: JSON.stringify(payload)
  });
}

export function validateProvider(payload: { base_url: string; api_key?: string; endpoint?: string; timeout_sec?: number }) {
  return jsonFetch<{ ok: boolean; detail: string }>('/api/settings/validate-provider', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function discoverMcpTools() {
  return jsonFetch<MCPDiscovery>('/api/settings/mcp/discover');
}

// --- Provider CRUD -------------------------------------------------------
export type ProviderCreateInput = {
  name: string;
  provider_name?: string;
  base_url: string;
  endpoint?: string;
  api_key?: string;
  request_timeout_sec?: number;
  enabled?: boolean;
  notes?: string;
  models?: Array<{
    name: string;
    display_name?: string;
    context_window_size?: number;
    max_output_tokens?: number;
    temperature?: number;
    top_p?: number;
    extra_params_json?: Record<string, any>;
    is_default?: boolean;
    enabled?: boolean;
  }>;
};

export type ModelEntryInput = {
  name: string;
  display_name?: string;
  context_window_size?: number;
  max_output_tokens?: number;
  temperature?: number;
  top_p?: number;
  extra_params_json?: Record<string, any>;
  is_default?: boolean;
  enabled?: boolean;
};

export function listProviders() {
  return jsonFetch<ProviderListResponse>('/api/providers');
}

export function createProvider(payload: ProviderCreateInput) {
  return jsonFetch<Provider>('/api/providers', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function updateProvider(
  providerId: string,
  payload: Partial<{
    name: string;
    provider_name: string;
    base_url: string;
    endpoint: string;
    api_key: string;
    request_timeout_sec: number;
    enabled: boolean;
    notes: string;
  }>
) {
  return jsonFetch<Provider>(`/api/providers/${providerId}`, {
    method: 'PATCH',
    body: JSON.stringify(payload)
  });
}

export function deleteProvider(providerId: string) {
  return jsonFetch<{ ok: boolean; deleted_provider_id: string }>(`/api/providers/${providerId}`, {
    method: 'DELETE'
  });
}

export function addProviderModel(providerId: string, payload: ModelEntryInput) {
  return jsonFetch<ModelEntry>(`/api/providers/${providerId}/models`, {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function updateProviderModel(
  providerId: string,
  modelId: string,
  payload: Partial<ModelEntryInput>
) {
  return jsonFetch<ModelEntry>(`/api/providers/${providerId}/models/${modelId}`, {
    method: 'PATCH',
    body: JSON.stringify(payload)
  });
}

export function deleteProviderModel(providerId: string, modelId: string) {
  return jsonFetch<{ ok: boolean; deleted_model_id: string }>(
    `/api/providers/${providerId}/models/${modelId}`,
    {
      method: 'DELETE'
    }
  );
}

export function activateProviderModel(providerId: string, modelId: string) {
  return jsonFetch<{
    ok: boolean;
    active_provider_id: string | null;
    active_model_id: string | null;
  }>(`/api/providers/${providerId}/models/${modelId}/activate`, {
    method: 'POST'
  });
}

export function setActiveSelection(payload: {
  provider_id: string | null;
  model_id?: string | null;
}) {
  return jsonFetch<{
    ok: boolean;
    active_provider_id: string | null;
    active_model_id: string | null;
  }>('/api/providers/active', {
    method: 'POST',
    body: JSON.stringify(payload)
  });
}

export function getSession(sessionId: string) {
  return jsonFetch<Session>(`/api/sessions/${sessionId}`);
}

export function getMemorySnapshot(sessionId: string) {
  return jsonFetch<any>(`/api/memory/${sessionId}/snapshot`);
}

export function startBackgroundRun(sessionId: string, content: string) {
  return jsonFetch<Run>(`/api/runs/${sessionId}`, {
    method: 'POST',
    body: JSON.stringify({ content })
  });
}

export function listRuns(sessionId: string) {
  return jsonFetch<Run[]>(`/api/runs/${sessionId}`);
}

export function cancelRun(sessionId: string, runId: string) {
  return jsonFetch<Run>(`/api/runs/${sessionId}/${runId}/cancel`, {
    method: 'POST'
  });
}

export function listRunEvents(sessionId: string, runId: string) {
  return jsonFetch<RunEvent[]>(`/api/runs/${sessionId}/${runId}/events`);
}

export function listRunArtifacts(sessionId: string, runId: string) {
  return jsonFetch<RunArtifact[]>(`/api/runs/${sessionId}/${runId}/artifacts`);
}

export function startEvolution(input: {
  prompt?: string;
  max_generations?: number;
  mode?: 'conservative' | 'experimental' | 'tests-only';
  stop_on_failure?: boolean;
}) {
  return jsonFetch<EvolutionRun>('/api/evolution/start', {
    method: 'POST',
    body: JSON.stringify(input)
  });
}

export function listEvolutionRuns() {
  return jsonFetch<EvolutionRun[]>('/api/evolution/runs');
}

export function listEvolutionEvents(runId: string) {
  return jsonFetch<EvolutionEvent[]>(`/api/evolution/runs/${runId}/events`);
}

export function cancelEvolutionRun(runId: string) {
  return jsonFetch<EvolutionRun>(`/api/evolution/runs/${runId}/cancel`, {
    method: 'POST'
  });
}

export function listEvolutionGenerations() {
  return jsonFetch<EvolutionGeneration[]>('/api/evolution/generations');
}

export function activateEvolutionGeneration(generation: number) {
  return jsonFetch<EvolutionGeneration>(`/api/evolution/generations/${generation}/activate`, {
    method: 'POST'
  });
}

export function copyEvolutionGenerationToRoot(generation: number) {
  return jsonFetch<{
    ok: boolean;
    generation: number;
    root_repo_path: string;
    source_repo_path: string;
    copied_at: string;
  }>(`/api/evolution/generations/${generation}/copy-to-root`, {
    method: 'POST'
  });
}

export function deleteEvolutionGeneration(generation: number, force: boolean = false) {
  const q = force ? '?force=true' : '';
  return jsonFetch<{ ok: boolean; deleted_generation: number; active_generation?: number | null }>(
    `/api/evolution/generations/${generation}${q}`,
    {
      method: 'DELETE'
    }
  );
}

export function runMemoryLint(sessionId: string, reason: string = 'manual') {
  const q = new URLSearchParams({ reason }).toString();
  return jsonFetch<any>(`/api/memory/${sessionId}/lint?${q}`, {
    method: 'POST'
  });
}

export async function streamChat(
  sessionId: string,
  content: string,
  thinkingMode: ThinkingModeTurn,
  onEvent: (event: StreamEvent) => void,
  options: {
    provider_id?: string | null;
    model_id?: string | null;
    force_search?: boolean;
    bypass_search_cache?: boolean;
    context_mode_override?: 'full' | 'skill_state' | null;
  } = {}
): Promise<void> {
  const res = await fetch(`${API_URL}/api/chat/${sessionId}/stream`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream'
    },
    body: JSON.stringify({
      content,
      thinking_mode: thinkingMode,
      provider_id: options.provider_id ?? undefined,
      model_id: options.model_id ?? undefined,
      force_search: options.force_search ?? false,
      bypass_search_cache: options.bypass_search_cache ?? false,
      context_mode_override: options.context_mode_override ?? null
    })
  });

  if (!res.ok || !res.body) {
    throw new Error(`Stream failed: ${res.status}`);
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });

    while (true) {
      const splitIndex = buffer.indexOf('\n\n');
      if (splitIndex === -1) {
        break;
      }
      const rawEvent = buffer.slice(0, splitIndex);
      buffer = buffer.slice(splitIndex + 2);

      const lines = rawEvent.split('\n');
      let eventName = 'message';
      let dataText = '';

      for (const line of lines) {
        if (line.startsWith('event:')) {
          eventName = line.slice(6).trim();
        }
        if (line.startsWith('data:')) {
          dataText += line.slice(5).trim();
        }
      }

      if (!dataText) {
        continue;
      }

      try {
        const data = JSON.parse(dataText);
        onEvent({ event: eventName, data });
      } catch {
        onEvent({ event: eventName, data: dataText });
      }
    }
  }
}
