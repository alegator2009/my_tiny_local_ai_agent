// Graph data structures shared between the SessionGraph UI and the backend
// /api/sessions/{id}/graph endpoint. Keep the field names in sync with
// apps/api/app/services/graph.py (build_session_graph).

export type GraphNodeType =
  | 'window'
  | 'checkpoint'
  | 'user'
  | 'assistant'
  | 'system'
  | 'tool_call'
  | 'tool_result'
  | 'mcp_tool_call'
  | 'mcp_tool_result'
  | 'terminal_command'
  | 'terminal_output'
  | 'file_change'
  | 'file_snapshot'
  | 'diff_summary'
  | 'test_result'
  | 'build_error'
  | 'image_created'
  | 'artifact_created';

export interface GraphNode {
  id: string;
  type: GraphNodeType;
  label: string;
  sub_label: string;
  window_id: string | null;
  turn_id: string | null;
  ts: string;
  token_count: number;
  meta: Record<string, unknown>;
}

export type GraphEdgeKind = 'sequence' | 'tool_io' | 'contains' | 'summarizes' | 'resumes' | 'closes';

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  kind: GraphEdgeKind;
  label: string;
}

export interface GraphWindow {
  id: string;
  index: number;
  started_at: string;
  closed_at: string | null;
  closing_reason: string | null;
  token_limit: number;
  rollover_trigger_percent: number;
  checkpoint_id: string | null;
}

export interface GraphCheckpoint {
  id: string;
  source_window_id: string;
  checkpoint_index: number;
  created_at: string;
  summary_text: string;
  decisions_count: number;
}

export interface SessionGraph {
  session_id: string;
  windows: GraphWindow[];
  checkpoints: GraphCheckpoint[];
  nodes: GraphNode[];
  edges: GraphEdge[];
}
