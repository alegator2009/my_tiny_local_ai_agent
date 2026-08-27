// Cognitive graph layout for the session.
//
// Strategy:
//   - All node types from the backend are first-class. Nothing is filtered out.
//   - Inside each window we build a TIME-ordered spine: every event (chat,
//     mcp call/result, file change, terminal, artifact, test, build error)
//     is placed on this spine in the order it occurred.
//   - Assistant / system messages sit on the spine. Tool calls/results
//     branch off as side leaves (below the assistant that triggered
//     them), so the graph shows the cognitive loop:
//       user -> assistant -> [mcp call] -> [mcp result] -> assistant
//   - Window "frames" sit in a topmost row, and the spine of the
//     first event in each window hangs off that frame.
//   - Checkpoints sit in their own row at the bottom; each window
//     connects to its checkpoint (if any), and a checkpoint connects
//     to the first event of the next window (resumes).
//   - Dagre (TB) handles the rest: it picks Y based on rank, X to keep
//     siblings apart. Because branches (tool leaves) share the same
//     top-to-bottom rank as their assistant, but go a bit further down,
//     the result is a tall, clearly branching tree — not a single line.

import dagre from 'dagre';
import type {
  GraphEdge,
  GraphNode,
  GraphNodeType,
  SessionGraph,
} from './graphTypes';

export interface TreeNode {
  id: string;
  x: number;
  y: number;
  r: number;
  type: GraphNodeType;
  /** Optional second-line label rendered in the tooltip. */
  label?: string;
  sub?: string;
  /** For zoom-aware rendering: true for "important" (non-chat) nodes
   *  that should be drawn as small labelled squares when zoomed in. */
  prominent: boolean;
}

export interface TreeLayoutResult {
  nodes: TreeNode[];
  width: number;
  height: number;
}

// Compact "atomic" radius for chat messages. Dagre is given a fixed cell
// size for these so the spine stays readable.
const R_CHAT = 5;
const R_LLM = 6.5; // assistant is a bit bigger - it's the cognitive core
const R_TOOL = 4.5;
const R_FILE = 4;
const R_CHECK = 6;
const R_WIN = 6.5;

const TYPE_RADIUS: Record<string, number> = {
  window: R_WIN,
  checkpoint: R_CHECK,
  user: R_CHAT,
  assistant: R_LLM,
  system: R_CHAT,
  tool_call: R_TOOL,
  tool_result: R_TOOL,
  mcp_tool_call: R_TOOL,
  mcp_tool_result: R_TOOL,
  terminal_command: R_TOOL,
  terminal_output: R_TOOL,
  file_change: R_FILE,
  file_snapshot: R_FILE,
  diff_summary: R_FILE,
  test_result: R_FILE,
  build_error: R_TOOL,
  image_created: R_FILE,
  artifact_created: R_FILE,
};

// "Prominent" types deserve their own labelled card at high zoom.
const PROMINENT_TYPES = new Set<string>([
  'mcp_tool_call',
  'mcp_tool_result',
  'tool_call',
  'tool_result',
  'terminal_command',
  'terminal_output',
  'file_change',
  'file_snapshot',
  'diff_summary',
  'test_result',
  'build_error',
  'image_created',
  'artifact_created',
  'checkpoint',
  'window',
]);

const CELL_W = 26;
const CELL_H = 22;

interface NodeSpec {
  id: string;
  type: GraphNodeType;
  r: number;
  prominent: boolean;
  label: string;
  sub: string;
}

function shortLabel(s: string | null | undefined, max = 28): string {
  if (!s) return '';
  const t = s.replace(/\s+/g, ' ').trim();
  return t.length > max ? t.slice(0, max - 1) + '…' : t;
}

function buildSpec(n: GraphNode): NodeSpec {
  const type = n.type;
  return {
    id: n.id,
    type,
    r: TYPE_RADIUS[type] ?? R_CHAT,
    prominent: PROMINENT_TYPES.has(type),
    label: n.label || n.id,
    sub: n.sub_label || '',
  };
}

export function layoutTree(
  graph: SessionGraph,
  options: { width: number; height: number }
): TreeLayoutResult {
  const g = new dagre.graphlib.Graph();
  g.setGraph({
    rankdir: 'TB',
    nodesep: 18,
    edgesep: 12,
    ranksep: 60,
    marginx: 24,
    marginy: 24,
  });
  g.setDefaultEdgeLabel(() => ({}));

  const specs: NodeSpec[] = [];
  const specById = new Map<string, NodeSpec>();

  // 1) Window trunk anchors (one per window) - topmost row.
  const orderedWindows = [...graph.windows].sort((a, b) =>
    a.started_at < b.started_at ? -1 : 1
  );
  for (const w of orderedWindows) {
    const s: NodeSpec = {
      id: `__win__:${w.id}`,
      type: 'window',
      r: R_WIN,
      prominent: true,
      label: `Window #${w.index}`,
      sub: w.closing_reason || 'active',
    };
    specs.push(s);
    specById.set(s.id, s);
  }

  // 2) Checkpoint nodes - bottom row.
  for (const cp of graph.checkpoints) {
    const s: NodeSpec = {
      id: `__cp__:${cp.id}`,
      type: 'checkpoint',
      r: R_CHECK,
      prominent: true,
      label: `Checkpoint #${cp.checkpoint_index}`,
      sub: shortLabel(cp.summary_text, 36),
    };
    specs.push(s);
    specById.set(s.id, s);
  }

  // 3) Real graph nodes from the backend.
  for (const n of graph.nodes) {
    const s = buildSpec(n);
    specs.push(s);
    specById.set(s.id, s);
  }

  for (const s of specs) {
    g.setNode(s.id, { width: CELL_W, height: CELL_H });
  }

  // ----- Group events per window, time-ordered. -----
  // We treat both messages AND workspace_events as part of the spine.
  interface TimelineItem {
    id: string; // node id
    ts: string;
    windowId: string | null;
  }
  const timelineByWindow = new Map<string, TimelineItem[]>();
  for (const n of graph.nodes) {
    if (n.type === 'window' || n.type === 'checkpoint') continue;
    const wid = n.window_id ?? '_orphan';
    const arr = timelineByWindow.get(wid) ?? [];
    arr.push({ id: n.id, ts: n.ts, windowId: wid });
    timelineByWindow.set(wid, arr);
  }
  for (const arr of timelineByWindow.values()) {
    arr.sort((a, b) =>
      a.ts < b.ts ? -1 : a.ts > b.ts ? 1 : a.id.localeCompare(b.id)
    );
  }

  // ----- 4) Spine edges: connect ONLY chat messages in time order.
  // Non-chat nodes (tool_call, tool_result, mcp_*, file_*, terminal_*,
  // diff_*, test_*, build_error, image_created, artifact_created) are
  // treated as side branches and connected to their assistant parent
  // in step 7. The result is a clear cognitive loop: user -> assistant
  // -> assistant -> ... with tool/IO leaves branching off below.
  const chatOnlyTypes = new Set<string>([
    'user',
    'assistant',
    'system',
  ]);
  for (const [, items] of timelineByWindow) {
    let prevChat: string | null = null;
    for (const it of items) {
      const t = specById.get(it.id)?.type;
      if (!t) continue;
      if (!chatOnlyTypes.has(t)) {
        // Side branch - skip; will be wired up in step 7.
        continue;
      }
      if (prevChat !== null) {
        g.setEdge(prevChat, it.id);
      }
      prevChat = it.id;
    }
  }

  // ----- 5) Window trunk -> first event of that window. -----
  for (const w of orderedWindows) {
    const items = timelineByWindow.get(w.id) ?? [];
    if (items.length > 0) {
      g.setEdge(`__win__:${w.id}`, items[0].id);
    }
  }

  // ----- 6) Window -> next window's first event (continuity). -----
  for (let i = 0; i < orderedWindows.length - 1; i++) {
    const nextItems = timelineByWindow.get(orderedWindows[i + 1].id) ?? [];
    if (nextItems.length === 0) continue;
    const myItems = timelineByWindow.get(orderedWindows[i].id) ?? [];
    if (myItems.length > 0) {
      // Last of this window -> first of next
      g.setEdge(myItems[myItems.length - 1].id, nextItems[0].id);
    } else {
      // Empty window: connect frame directly.
      g.setEdge(
        `__win__:${orderedWindows[i].id}`,
        nextItems[0].id
      );
    }
  }

  // ----- 7) Side branches: assistant -> tool/mcp leaves in same turn. -----
  // We do this BEFORE step 8 so the tool_io edges don't fight with the
  // spine. Assistant is the parent; child is the tool/result/leaf node.
  // In TB layout the parent sits above the leaves; the ranksep pushes
  // siblings down so branches stay readable.
  const turnsToAssistant = new Map<string, string>(); // turn_id -> assistant node id
  for (const n of graph.nodes) {
    if (n.type === 'assistant' && n.turn_id) {
      // First assistant wins per turn.
      if (!turnsToAssistant.has(n.turn_id)) {
        turnsToAssistant.set(n.turn_id, n.id);
      }
    }
  }

  const sideBranchTypes = new Set<string>([
    'tool_call',
    'tool_result',
    'mcp_tool_call',
    'mcp_tool_result',
    'terminal_command',
    'terminal_output',
    'file_change',
    'file_snapshot',
    'diff_summary',
    'test_result',
    'build_error',
    'image_created',
    'artifact_created',
  ]);

  // Map from node id to its window_id for fast lookup.
  const widByNodeId = new Map<string, string>();
  for (const n of graph.nodes) {
    if (n.window_id) widByNodeId.set(n.id, n.window_id);
  }
  const tsByNodeId = new Map<string, string>();
  for (const n of graph.nodes) tsByNodeId.set(n.id, n.ts);

  // Helper: find the most recent assistant in the same window whose ts
  // is <= node.ts. That's the cognitive "parent" for the side branch.
  const findParentAssistant = (nodeId: string): string | null => {
    const wid = widByNodeId.get(nodeId);
    const ts = tsByNodeId.get(nodeId);
    if (!wid || !ts) return null;
    const items = (timelineByWindow.get(wid) ?? []).filter(
      (it) => it.id !== nodeId && tsByNodeId.get(it.id) && tsByNodeId.get(it.id)! <= ts
    );
    if (items.length === 0) return null;
    // Walk backwards through the spine to find the most recent assistant.
    for (let i = items.length - 1; i >= 0; i--) {
      const cand = items[i];
      if (specById.get(cand.id)?.type === 'assistant') {
        return cand.id;
      }
    }
    return null;
  };

  for (const n of graph.nodes) {
    if (!sideBranchTypes.has(n.type)) continue;
    // If the previous item on the spine is the assistant, the spine
    // already gives us the parent->child connection. But since we
    // explicitly EXCLUDED non-chat nodes from the spine in step 4,
    // we now always need to branch these off. Side branches use
    // weight=1 so dagre can put them on a different rank below.
    const wid = n.window_id;
    if (!wid) continue;
    const parent = findParentAssistant(n.id);
    if (parent && parent !== n.id) {
      g.setEdge(parent, n.id, { weight: 1 });
    }
  }

  // ----- 7b) Tool call -> tool result as a SHORT edge inside the
  //           same branch. Both call and result hang off the same
  //           assistant. We attach the result to the call with a
  //           weight so dagre keeps them on adjacent ranks below
  //           the assistant. The call is itself attached to the
  //           assistant with weight 1 in step 7. -----
  // We need to do this AFTER step 7 so the assistant->call edges
  // exist; otherwise dagre creates long routing around.
  // (We re-loop here; finding the assistant parent for the call and
  // attaching the result to the call directly.)
  const resultToCall = new Map<string, string>(); // resultId -> callId
  for (const [wid, items] of timelineByWindow) {
    let openCall: string | null = null;
    for (const it of items) {
      const t = specById.get(it.id)?.type;
      if (t === 'tool_call' || t === 'mcp_tool_call') {
        openCall = it.id;
      } else if (t === 'tool_result' || t === 'mcp_tool_result') {
        if (openCall) {
          resultToCall.set(it.id, openCall);
          openCall = null;
        }
      }
    }
  }
  for (const [resultId, callId] of resultToCall) {
    g.setEdge(callId, resultId, { weight: 1 });
  }

  // ----- 8) Backend-declared edges (tool_io, contains, summarizes,
  //           resumes, closes). We only honor tool_io if both endpoints
  //           exist; we always honor contains/summarizes/resumes/closes. -----
  for (const e of graph.edges) {
    if (e.kind === 'sequence') {
      // Already drawn by step 4.
      continue;
    }
    if (e.kind === 'contains' || e.kind === 'summarizes' || e.kind === 'resumes' || e.kind === 'closes') {
      if (specById.has(e.source) && specById.has(e.target) && e.source !== e.target) {
        g.setEdge(e.source, e.target);
      }
    } else if (e.kind === 'tool_io') {
      if (specById.has(e.source) && specById.has(e.target) && e.source !== e.target) {
        g.setEdge(e.source, e.target);
      }
    }
  }

  // ----- 9) Window -> checkpoint edge (if backend didn't declare one). -----
  for (const w of orderedWindows) {
    if (!w.checkpoint_id) continue;
    const winId = `__win__:${w.id}`;
    const cpId = `__cp__:${w.checkpoint_id}`;
    if (specById.has(winId) && specById.has(cpId)) {
      // Already added by step 8 if backend declared "summarizes". Add
      // anyway to be safe - dagre is fine with redundant edges.
      g.setEdge(winId, cpId);
    }
  }

  dagre.layout(g);

  const out: TreeNode[] = [];
  for (const s of specs) {
    const lbl = g.node(s.id) as
      | { x?: number; y?: number }
      | undefined;
    if (!lbl) continue;
    const x = typeof lbl.x === 'number' ? lbl.x : 0;
    const y = typeof lbl.y === 'number' ? lbl.y : 0;
    out.push({
      id: s.id,
      x,
      y,
      r: s.r,
      type: s.type,
      label: s.label,
      sub: s.sub,
      prominent: s.prominent,
    });
  }

  // Bounding box + shift to (PAD, PAD).
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity;
  for (const n of out) {
    if (n.x - n.r < minX) minX = n.x - n.r;
    if (n.y - n.r < minY) minY = n.y - n.r;
    if (n.x + n.r > maxX) maxX = n.x + n.r;
    if (n.y + n.r > maxY) maxY = n.y + n.r;
  }
  if (!isFinite(minX)) {
    minX = 0;
    minY = 0;
    maxX = options.width;
    maxY = options.height;
  }
  const PAD = 24;
  const shiftX = -minX + PAD;
  const shiftY = -minY + PAD;
  for (const n of out) {
    n.x += shiftX;
    n.y += shiftY;
  }
  const width = Math.max(options.width, maxX - minX + PAD * 2);
  const height = Math.max(options.height, maxY - minY + PAD * 2);

  return { nodes: out, width, height };
}
