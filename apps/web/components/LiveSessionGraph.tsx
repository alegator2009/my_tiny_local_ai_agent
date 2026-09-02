'use client';

import {
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  Handle,
  Position,
  type Node,
  type Edge,
  type EdgeProps,
  type NodeProps,
  type Viewport,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';

import { useCallback, useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';

import { getSessionGraph, type WindowState } from '@/lib/api';
import type { GraphNode as GraphNodeData, SessionGraph } from '@/lib/graphTypes';
import { layoutTree, type TreeNode } from '@/lib/treeLayout';

interface Props {
  sessionId: string | null;
  width: number;
  height: number;
  inFlight: Set<string>;
  lastEventAt: Date | null;
  windowState: WindowState | null;
}

/**
 * Interpolate a context-percent (0..1) into a stroke colour. 0% = the
 * default phosphor green, 100% = dark red. We stay roughly within the
 * Matrix palette so the graph still feels native, but the hue rotates
 * green -> amber -> red so that a high-context session is immediately
 * obvious from a distance.
 */
function contextStroke(percent: number, alpha = 0.85): string {
  const p = Math.max(0, Math.min(1, percent));
  // Stops:
  //   0.00 -> (0, 255, 102)   matrix green
  //   0.55 -> (255, 200, 60)  amber
  //   0.80 -> (255, 110, 50)  orange
  //   1.00 -> (170, 0, 0)     dark red
  let r: number, g: number, b: number;
  if (p < 0.55) {
    const t = p / 0.55;
    r = Math.round(0 + (255 - 0) * t);
    g = Math.round(255 + (200 - 255) * t);
    b = Math.round(102 + (60 - 102) * t);
  } else if (p < 0.8) {
    const t = (p - 0.55) / 0.25;
    r = Math.round(255 + (255 - 255) * t);
    g = Math.round(200 + (110 - 200) * t);
    b = Math.round(60 + (50 - 60) * t);
  } else {
    const t = (p - 0.8) / 0.2;
    r = Math.round(255 + (170 - 255) * t);
    g = Math.round(110 + (0 - 110) * t);
    b = Math.round(50 + (0 - 50) * t);
  }
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

const TYPE_HATCH: Record<string, string> = {
  window: 'solid',
  checkpoint: 'cross',
  user: 'ring',
  assistant: 'dot',
  tool_call: 'diag',
  tool_result: 'diag-back',
  mcp_tool_call: 'diag',
  mcp_tool_result: 'diag-back',
  terminal_command: 'vert',
  terminal_output: 'horiz',
  file_change: 'tl-br',
  file_snapshot: 'tl-br',
  diff_summary: 'grid',
  test_result: 'grid',
  build_error: 'cross',
  image_created: 'ring',
  artifact_created: 'dot',
  system: 'solid',
};

// Cognitive-category accent. Each "non-chat" node type maps to a
// short tag (a single uppercase letter) shown inside its card at high
// zoom. This makes the cognitive architecture immediately readable.
const CATEGORY_TAG: Record<string, string> = {
  window: 'W',
  checkpoint: 'CK',
  assistant: 'LLM',
  user: 'U',
  system: 'SYS',
  tool_call: 'TC',
  tool_result: 'TR',
  mcp_tool_call: 'MCP',
  mcp_tool_result: 'MCP',
  terminal_command: 'SH',
  terminal_output: 'OUT',
  file_change: 'FILE',
  file_snapshot: 'FILE',
  diff_summary: 'DIFF',
  test_result: 'TEST',
  build_error: 'ERR',
  image_created: 'IMG',
  artifact_created: 'ART',
};

// Outline accent color for prominent node types. Matrix-green stays
// the default for everything, but the *outline* of prominent nodes
// gets a category-specific color so the cognitive loop is visible at
// a glance.
const CATEGORY_COLOR: Record<string, string> = {
  mcp_tool_call: 'rgba(177, 78, 255, 0.95)',     // purple
  mcp_tool_result: 'rgba(177, 78, 255, 0.65)',
  tool_call: 'rgba(0, 212, 255, 0.95)',         // cyan
  tool_result: 'rgba(0, 212, 255, 0.6)',
  terminal_command: 'rgba(255, 186, 92, 0.95)', // amber
  terminal_output: 'rgba(255, 186, 92, 0.55)',
  file_change: 'rgba(255, 222, 92, 0.95)',      // yellow
  file_snapshot: 'rgba(255, 222, 92, 0.6)',
  diff_summary: 'rgba(255, 222, 92, 0.85)',
  test_result: 'rgba(135, 217, 154, 0.95)',     // green
  build_error: 'rgba(255, 95, 95, 0.95)',       // red
  image_created: 'rgba(255, 137, 222, 0.95)',   // pink
  artifact_created: 'rgba(255, 137, 222, 0.85)',
  checkpoint: 'rgba(0, 212, 255, 0.95)',
  window: 'rgba(0, 255, 156, 0.95)',
};

interface DotData extends Record<string, unknown> {
  radius: number;
  type: string;
  label: string;
  sub: string;
  inFlight: boolean;
  hatch: string;
  hover: boolean;
  prominent: boolean;
  tag: string;
  accent: string;
  appear: boolean;
  stagger: string;
}

function DotNode({ data }: NodeProps) {
  const d = data as unknown as DotData;
  const r = d.radius;
  const size = r * 2;
  const hatchClass = `lg-dot lg-hatch-${d.hatch}`;
  return (
    <div
      className={hatchClass}
      style={{ width: size, height: size }}
      data-inflight={d.inFlight ? '1' : '0'}
      data-hover={d.hover ? '1' : '0'}
      data-appear={d.appear ? '1' : '0'}
      data-i={d.stagger || '0'}
    >
      <Handle
        type="target"
        position={Position.Top}
        id="t"
        isConnectable={false}
        style={{ background: 'transparent', border: 'none', width: 1, height: 1, opacity: 0.001 }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="b"
        isConnectable={false}
        style={{ background: 'transparent', border: 'none', width: 1, height: 1, opacity: 0.001 }}
      />
    </div>
  );
}

// CardNode: rendered for "prominent" types (mcp, file, artifact,
// checkpoint, window). Sits on top of the same dot anchor, but the
// user perceives a labelled card at high zoom and a colored dot at
// low zoom (we use scale to keep both visuals consistent).
function CardNode({ data }: NodeProps) {
  const d = data as unknown as DotData;
  const w = d.radius * 4;
  const h = d.radius * 2.6;
  const accent = d.accent || 'rgba(0, 255, 156, 0.95)';
  return (
    <div
      className="lg-card"
      style={{
        width: w,
        height: h,
        borderColor: accent,
        boxShadow: `0 0 0 1px ${accent}, 0 0 8px ${accent}55`,
      }}
      data-inflight={d.inFlight ? '1' : '0'}
      data-hover={d.hover ? '1' : '0'}
      data-appear={d.appear ? '1' : '0'}
    >
      <span className="lg-card-tag" style={{ color: accent, borderColor: accent }}>
        {d.tag}
      </span>
      <span className="lg-card-label">{d.label}</span>
      <Handle
        type="target"
        position={Position.Top}
        id="t"
        isConnectable={false}
        style={{ background: 'transparent', border: 'none', width: 1, height: 1, opacity: 0.001 }}
      />
      <Handle
        type="source"
        position={Position.Bottom}
        id="b"
        isConnectable={false}
        style={{ background: 'transparent', border: 'none', width: 1, height: 1, opacity: 0.001 }}
      />
    </div>
  );
}

const NODE_TYPES = { dot: DotNode, card: CardNode };

interface EdgeData extends Record<string, unknown> {
  appear: boolean;
  hover: boolean;
  contextPercent: number;
  contextUsedTokens: number;
  contextTokenLimit: number;
}

/**
 * Custom edge component. ReactFlow v12 wraps the path inside
 * `.react-flow__edge`; we set `data-appear` and `data-hover` on the
 * wrapper so the CSS selectors we wrote in globals.css can reach the
 * underlying path.
 */
function ContextEdge(props: EdgeProps) {
  const { sourceX, sourceY, targetX, targetY, style, data, markerEnd } = props;
  const d = data as EdgeData | undefined;
  const dAppear = d?.appear ? '1' : '0';
  const dHover = d?.hover ? '1' : '0';
  return (
    <g className="react-flow__edge-context" data-appear={dAppear} data-hover={dHover}>
      <path
        className="react-flow__edge-path"
        d={`M ${sourceX},${sourceY} L ${targetX},${targetY}`}
        style={style}
        markerEnd={markerEnd}
      />
    </g>
  );
}

const EDGE_TYPES = { context: ContextEdge };

function LiveGraphInner({ sessionId, width, height, inFlight, lastEventAt, windowState }: Props) {
  const [graph, setGraph] = useState<SessionGraph | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [hovered, setHovered] = useState<string | null>(null);

  const graphRef = useRef<SessionGraph | null>(null);
  const positionsRef = useRef<Map<string, { x: number; y: number }>>(new Map());

  // First-time we see an id, we mark it "new" so the CSS can run the
  // appear animation; after ~700ms the flag is cleared so subsequent
  // re-renders are quiet.
  const seenNodeIdsRef = useRef<Set<string>>(new Set());
  const seenEdgeIdsRef = useRef<Set<string>>(new Set());
  const appearNodeIdsRef = useRef<Set<string>>(new Set());
  const appearEdgeIdsRef = useRef<Set<string>>(new Set());
  const appearTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Edge hover state (separate from node hover)
  const [hoveredEdge, setHoveredEdge] = useState<{
    id: string;
    screen: { x: number; y: number };
  } | null>(null);

  const reactFlow = useReactFlow();
  const viewportRef = useRef<Viewport | null>(null);

  // Initial load: only when sessionId changes. Subsequent updates come from
  // the lastEventAt watcher, which merges new events into the existing graph
  // (positions and edges are preserved).
  useEffect(() => {
    if (!sessionId) {
      graphRef.current = null;
      positionsRef.current.clear();
      setGraph(null);
      return;
    }
    let cancelled = false;
    setError(null);
    // First load only if we don't already have data for this session
    if (graphRef.current?.session_id !== sessionId) {
      positionsRef.current.clear();
      getSessionGraph(sessionId)
        .then((data) => {
          if (cancelled) return;
          graphRef.current = data;
          setGraph(data);
        })
        .catch((e) => {
          if (cancelled) return;
          setError(e instanceof Error ? e.message : String(e));
        });
    }
    return () => {
      cancelled = true;
    };
  }, [sessionId]);

  // Incremental refresh: when lastEventAt changes (i.e. a new event arrived),
  // refetch the graph and merge it. We keep the existing positions for nodes
  // we already know about, only laying out newly-discovered nodes.
  const lastSeenRef = useRef<number>(0);
  useEffect(() => {
    if (!sessionId || !lastEventAt) return;
    const stamp = lastEventAt.getTime();
    if (stamp <= lastSeenRef.current) return;
    lastSeenRef.current = stamp;
    // Throttle: skip if we just refreshed in the last 500ms
    if (Date.now() - (lastFetchRef.current ?? 0) < 500) return;
    lastFetchRef.current = Date.now();
    getSessionGraph(sessionId)
      .then((fresh) => {
        graphRef.current = fresh;
        setGraph(fresh);
      })
      .catch(() => {
        // ignore — keep previous graph
      });
  }, [lastEventAt, sessionId]);
  const lastFetchRef = useRef<number>(0);

  const layout = useMemo(() => {
    if (!graph) return null;
    const t = layoutTree(graph, { width, height });
    // Preserve positions for nodes that already had a known position so the
    // graph doesn't jump on every refresh.
    const preserved = positionsRef.current;
    const prev = new Map<string, { x: number; y: number }>();
    for (const n of t.nodes) {
      if (preserved.has(n.id)) {
        prev.set(n.id, preserved.get(n.id)!);
      }
    }
    // For new nodes (not seen before), place them at the layout position.
    // For previously-seen nodes whose position was *not* overridden by the
    // user (we don't track that here), use the layout position. The
    // positionsRef cache is therefore a "where to draw" cache, and we
    // update it every render so dragging keeps the new position.
    for (const n of t.nodes) {
      if (!preserved.has(n.id)) {
        preserved.set(n.id, { x: n.x, y: n.y });
      }
    }
    return t;
  }, [graph, width, height]);

  const { nodes, edges, hoveredData, hoverScreen, contextInfo } = useMemo<{
    nodes: Node[];
    edges: Edge[];
    hoveredData: GraphNodeData | null;
    hoverScreen: { x: number; y: number } | null;
    contextInfo: { percent: number; usedTokens: number; tokenLimit: number };
  }>(() => {
    if (!graph || !layout) {
      return {
        nodes: [],
        edges: [],
        hoveredData: null,
        hoverScreen: null,
        contextInfo: { percent: 0, usedTokens: 0, tokenLimit: 0 },
      };
    }
    const posById = positionsRef.current;
    // Map live-hint synthetic ids to the type they should pulse. The
    // renderer keeps the rest of the graph stable; only the matching
    // category flashes.
    const liveHintType = (() => {
      for (const id of inFlight) {
        if (id === '__live__:llm') return 'assistant';
        if (id === '__live__:memory') return 'checkpoint';
        if (id === '__live__:checkpoint') return 'checkpoint';
      }
      return null;
    })();
    // Context fill from the current window state. 0..1 — clamped here
    // so a transient burst above 100% (e.g. pre-rollover) doesn't crash
    // the colour math.
    const percentRaw = windowState?.used_percent ?? 0;
    const percent = Math.max(0, Math.min(1, percentRaw / 100));
    const usedTokens = windowState?.used_tokens ?? 0;
    const tokenLimit = windowState?.token_limit ?? 0;

    // Diff: any node id we have NOT seen before is "new" and gets a
    // data-appear flag. Same for edges. The flag stays for ~700ms and
    // is then cleared (handled in a separate effect).
    const seenN = seenNodeIdsRef.current;
    const seenE = seenEdgeIdsRef.current;
    const appearN = appearNodeIdsRef.current;
    const appearE = appearEdgeIdsRef.current;
    for (const tn of layout.nodes) {
      if (!seenN.has(tn.id)) {
        seenN.add(tn.id);
        appearN.add(tn.id);
      }
    }
    for (const e of graph.edges) {
      if (!seenE.has(e.id)) {
        seenE.add(e.id);
        appearE.add(e.id);
      }
    }

    // Pre-compute a stable stagger index for each node based on its id
    // so neighbouring dots don't all bob in unison.
    const staggerIndex = (id: string): string => {
      let h = 0;
      for (let i = 0; i < id.length; i++) h = (h * 31 + id.charCodeAt(i)) >>> 0;
      return String(h % 8);
    };

    const xyNodes: Node[] = layout.nodes.map((tn) => {
      const p = posById.get(tn.id) ?? { x: tn.x, y: tn.y };
      const r = tn.r;
      // Prominent types get a small labelled card (rendered via
      // CardNode). Width/height of the card are 4r x 2.6r so we
      // anchor the position at the card's top-left.
      const isProminent = tn.prominent;
      const w = isProminent ? r * 4 : r * 2;
      const h = isProminent ? r * 2.6 : r * 2;
      const isLive = inFlight.has(tn.id) || (liveHintType !== null && tn.type === liveHintType);
      const isNew = appearN.has(tn.id);
      return {
        id: tn.id,
        type: isProminent ? 'card' : 'dot',
        position: { x: p.x - w / 2, y: p.y - h / 2 },
        // Tell ReactFlow the size up front so it never enters the
        // "needs measurement" state and never applies `visibility: hidden`.
        width: w,
        height: h,
        measured: { width: w, height: h },
        data: {
          radius: r,
          type: tn.type,
          label: tn.label || tn.id,
          sub: tn.sub || '',
          inFlight: isLive,
          hatch: TYPE_HATCH[tn.type] ?? 'solid',
          hover: hovered === tn.id,
          prominent: isProminent,
          tag: CATEGORY_TAG[tn.type] ?? '',
          accent: CATEGORY_COLOR[tn.type] ?? 'rgba(0, 255, 156, 0.95)',
          // Render-time flags surfaced via data-attributes
          appear: isNew,
          stagger: staggerIndex(tn.id),
        },
        draggable: true,
        selectable: true,
      };
    });
    const baseStroke = contextStroke(percent, 0.85);
    const xyEdges: Edge[] = graph.edges.map((e) => {
      const isTool = e.kind === 'tool_io';
      const isContains = e.kind === 'contains';
      const isSummarizes = e.kind === 'summarizes' || e.kind === 'resumes' || e.kind === 'closes';
      // Tool-io edges keep a high-saturation green so the live tool
      // activity is still distinguishable; contains-edges are dimmed
      // and stay the base colour (they're frame-level and shouldn't
      // shout). Everything else takes the context colour.
      const stroke = isTool
        ? 'rgba(0, 255, 102, 0.95)'
        : isContains
        ? 'rgba(255, 255, 255, 0.18)'
        : isSummarizes
        ? contextStroke(percent, 0.95)
        : baseStroke;
      const strokeWidth = isTool ? 1.3 : isContains ? 0.6 : 0.95;
      const isNew = appearE.has(e.id);
      const isHover = hoveredEdge?.id === e.id;
      return {
        id: e.id,
        source: e.source,
        target: e.target,
        sourceHandle: 'b',
        targetHandle: 't',
        type: 'context',
        data: {
          appear: isNew,
          hover: isHover,
          contextPercent: percent,
          contextUsedTokens: usedTokens,
          contextTokenLimit: tokenLimit,
        },
        style: {
          stroke,
          strokeWidth: isHover ? Math.max(strokeWidth, 1.6) : strokeWidth,
          strokeDasharray: isContains ? '2 3' : undefined,
          color: stroke,
        },
        animated: isTool && inFlight.has(e.source),
      };
    });
    let hoveredData: GraphNodeData | null = null;
    let hoverScreen: { x: number; y: number } | null = null;
    if (hovered) {
      hoveredData = graph.nodes.find((n) => n.id === hovered) ?? null;
      const p = posById.get(hovered);
      if (p) hoverScreen = p;
    }
    return {
      nodes: xyNodes,
      edges: xyEdges,
      hoveredData,
      hoverScreen,
      contextInfo: { percent, usedTokens, tokenLimit },
    };
  }, [graph, layout, inFlight, hovered, hoveredEdge, windowState]);

  // Clear "appear" flags after the CSS animation has had time to play.
  // This makes sure the animation runs only once per id and not on every
  // re-render. The size of the appear sets is sampled into a state
  // value so the effect re-runs whenever something new arrived.
  const [appearTick, setAppearTick] = useState(0);
  useEffect(() => {
    setAppearTick((appearNodeIdsRef.current.size || 0) + (appearEdgeIdsRef.current.size || 0));
  }, [graph]);
  useEffect(() => {
    if (appearTick === 0) return;
    const t = setTimeout(() => {
      appearNodeIdsRef.current.clear();
      appearEdgeIdsRef.current.clear();
      // Force a re-render so the data-appear flag flips to 0.
      setGraph((g) => (g ? { ...g } : g));
    }, 800);
    return () => clearTimeout(t);
  }, [appearTick]);

  // Fit view once per session (and on size change if user hasn't interacted)
  const fittedRef = useRef<string | null>(null);
  useEffect(() => {
    if (nodes.length === 0) return;
    if (fittedRef.current === sessionId) return;
    const t = setTimeout(() => {
      try {
        reactFlow.fitView({ padding: 0.15, duration: 350 });
      } catch {
        // ignore
      }
      fittedRef.current = sessionId;
    }, 200);
    return () => clearTimeout(t);
  }, [sessionId, nodes.length, reactFlow]);

  const onMove = useCallback((_: unknown, viewport: Viewport) => {
    viewportRef.current = viewport;
  }, []);

  const onNodeDrag = useCallback((_: unknown, n: Node) => {
    const r = (n.data as DotData).radius;
    positionsRef.current.set(n.id, { x: n.position.x + r, y: n.position.y + r });
  }, []);

  const onNodeMouseEnter = useCallback((_: unknown, n: Node) => {
    setHovered(n.id);
  }, []);

  const onNodeMouseLeave = useCallback(() => {
    setHovered(null);
  }, []);

  // Edge hover handlers. The event we get from ReactFlow includes the
  // mouse coordinates already; we use them to anchor the tooltip
  // exactly where the cursor is, so the user always sees a tooltip
  // next to the edge they're pointing at.
  const onEdgeMouseEnter = useCallback(
    (event: MouseEvent, edge: Edge) => {
      const container = (event.currentTarget as HTMLElement | null)?.closest('.lg-canvas');
      const rect = (container as HTMLElement | null)?.getBoundingClientRect();
      if (!rect) return;
      setHoveredEdge({
        id: edge.id,
        screen: { x: event.clientX - rect.left, y: event.clientY - rect.top },
      });
    },
    []
  );

  const onEdgeMouseMove = useCallback(
    (event: MouseEvent, _edge: Edge) => {
      const container = (event.currentTarget as HTMLElement | null)?.closest('.lg-canvas');
      const rect = (container as HTMLElement | null)?.getBoundingClientRect();
      if (!rect) return;
      setHoveredEdge((prev) =>
        prev
          ? { ...prev, screen: { x: event.clientX - rect.left, y: event.clientY - rect.top } }
          : prev
      );
    },
    []
  );

  const onEdgeMouseLeave = useCallback(() => {
    setHoveredEdge(null);
  }, []);

  if (!sessionId) return <div className="lg-empty">Select a session to see its graph</div>;
  if (error) return <div className="lg-error">Graph error: {error}</div>;
  if (!graph) return <div className="lg-empty">Loading…</div>;

  return (
    <div className="lg-root" style={{ width, height }}>
      <div className="lg-header">
        <span className="lg-title">Graph</span>
        <span className="lg-stats">
          {graph.nodes.length} nodes · {graph.edges.length} edges
        </span>
        {lastEventAt ? (
          <span className="lg-pulse" suppressHydrationWarning title={`Last event: ${lastEventAt.toLocaleTimeString()}`}>
            ● live
          </span>
        ) : null}
      </div>
      <div className="lg-canvas">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={NODE_TYPES}
          edgeTypes={EDGE_TYPES}
          onNodeDrag={onNodeDrag}
          onNodeMouseEnter={onNodeMouseEnter}
          onNodeMouseLeave={onNodeMouseLeave}
          onEdgeMouseEnter={onEdgeMouseEnter}
          onEdgeMouseMove={onEdgeMouseMove}
          onEdgeMouseLeave={onEdgeMouseLeave}
          onMove={onMove}
          fitView
          proOptions={{ hideAttribution: true }}
          nodesDraggable
          nodesConnectable={false}
          elementsSelectable
          panOnDrag
          zoomOnScroll
          zoomOnPinch
          minZoom={0.1}
          maxZoom={2.5}
        />
        {hoveredData && hoverScreen ? (
          <NodeTooltip
            node={hoveredData}
            inFlight={inFlight.has(hoveredData.id)}
            anchor={hoverScreen}
            viewport={viewportRef.current}
            containerSize={{ width, height }}
          />
        ) : null}
        {hoveredEdge ? (
          <EdgeTooltip
            edge={graph.edges.find((e) => e.id === hoveredEdge.id) ?? null}
            anchor={hoveredEdge.screen}
            contextInfo={contextInfo}
            containerSize={{ width, height }}
          />
        ) : null}
      </div>
    </div>
  );
}

interface TooltipProps {
  node: GraphNodeData;
  inFlight: boolean;
  anchor: { x: number; y: number };
  viewport: Viewport | null;
  containerSize: { width: number; height: number };
}

function NodeTooltip({ node, inFlight, anchor, viewport, containerSize }: TooltipProps) {
  const v = viewport ?? { x: 0, y: 0, zoom: 1 };
  const sx = anchor.x * v.zoom + v.x;
  const sy = anchor.y * v.zoom + v.y;
  const left = Math.min(containerSize.width - 240, Math.max(8, sx + 14));
  const top = Math.min(containerSize.height - 200, Math.max(8, sy + 14));
  const meta = (node.meta ?? {}) as Record<string, unknown>;
  const payload =
    meta && typeof meta === 'object' && 'payload' in meta
      ? ((meta as Record<string, unknown>).payload as Record<string, unknown> | undefined)
      : undefined;
  const summary = typeof meta?.summary === 'string' ? (meta.summary as string) : '';
  const role = typeof meta?.role === 'string' ? (meta.role as string) : '';
  const isAnchor = Boolean(meta?.is_anchor);
  const isPinned = Boolean(meta?.is_pinned);

  // Pull a small set of well-known payload fields for the common types.
  const toolName =
    (payload && typeof payload.tool === 'string' && (payload.tool as string)) || '';
  const command =
    (payload && typeof payload.command === 'string' && (payload.command as string)) || '';
  const exitCode =
    payload && typeof payload.exit_code === 'number' ? (payload.exit_code as number) : null;
  const path =
    (payload && typeof payload.path === 'string' && (payload.path as string)) || '';
  const mime =
    (payload && typeof payload.mime_type === 'string' && (payload.mime_type as string)) || '';
  const sizeBytes =
    payload && typeof payload.size_bytes === 'number' ? (payload.size_bytes as number) : null;

  return (
    <div className="lg-tooltip" style={{ left, top }}>
      <div className="lg-tooltip-head">
        <span className="lg-tooltip-dot" />
        <span className="lg-tooltip-label">{node.label || node.id}</span>
        <span className="lg-tooltip-type">{node.type}</span>
      </div>
      {node.sub_label ? <div className="lg-tooltip-sub">{node.sub_label}</div> : null}
      {summary && summary !== node.sub_label ? (
        <div className="lg-tooltip-sub">{summary}</div>
      ) : null}
      {toolName ? (
        <div className="lg-tooltip-row">
          <span>tool</span>
          <code>{toolName}</code>
        </div>
      ) : null}
      {command ? (
        <div className="lg-tooltip-row">
          <span>cmd</span>
          <code>{command.length > 40 ? command.slice(0, 37) + '…' : command}</code>
        </div>
      ) : null}
      {exitCode !== null ? (
        <div className="lg-tooltip-row">
          <span>exit</span>
          <span style={{ color: exitCode === 0 ? '#87d99a' : '#ff9a9a' }}>{exitCode}</span>
        </div>
      ) : null}
      {path ? (
        <div className="lg-tooltip-row">
          <span>path</span>
          <code>{path.length > 40 ? path.slice(0, 37) + '…' : path}</code>
        </div>
      ) : null}
      {mime ? (
        <div className="lg-tooltip-row">
          <span>mime</span>
          <span>{mime}</span>
        </div>
      ) : null}
      {sizeBytes !== null ? (
        <div className="lg-tooltip-row">
          <span>size</span>
          <span>{sizeBytes < 1024 ? `${sizeBytes} B` : `${(sizeBytes / 1024).toFixed(1)} KB`}</span>
        </div>
      ) : null}
      {role ? (
        <div className="lg-tooltip-row">
          <span>role</span>
          <span>{role}</span>
        </div>
      ) : null}
      {isAnchor || isPinned ? (
        <div className="lg-tooltip-row">
          <span>flags</span>
          <span>
            {isAnchor ? '⚓ ' : ''}
            {isPinned ? '📌 ' : ''}
          </span>
        </div>
      ) : null}
      {node.token_count > 0 ? (
        <div className="lg-tooltip-row">
          <span>tokens</span>
          <span>{node.token_count}</span>
        </div>
      ) : null}
      <div className="lg-tooltip-row">
        <span>id</span>
        <code>{node.id}</code>
      </div>
      <div className="lg-tooltip-row">
        <span>status</span>
        <span className={inFlight ? 'lg-status-on' : 'lg-status-off'}>
          {inFlight ? 'firing' : 'idle'}
        </span>
      </div>
    </div>
  );
}

/**
 * Edge tooltip: shows the current context window fill (used / limit /
 * percent) plus a small colour bar matching the edge colour. The bar
 * is the same colour as the line so the user can connect the two at
 * a glance.
 */
function EdgeTooltip({
  edge,
  anchor,
  contextInfo,
  containerSize,
}: {
  edge: { id: string; kind: string; label: string; source: string; target: string } | null;
  anchor: { x: number; y: number };
  contextInfo: { percent: number; usedTokens: number; tokenLimit: number };
  containerSize: { width: number; height: number };
}) {
  const left = Math.min(containerSize.width - 220, Math.max(8, anchor.x + 12));
  const top = Math.min(containerSize.height - 120, Math.max(8, anchor.y + 12));
  const pct = contextInfo.percent;
  const pctStr = `${(pct * 100).toFixed(1)}%`;
  const used = contextInfo.usedTokens;
  const limit = contextInfo.tokenLimit;
  const fillColour = contextStroke(pct, 0.95);
  return (
    <div className="lg-edge-tooltip" style={{ left, top }}>
      <div className="lg-edge-tooltip-head">
        <span>Context @ this moment</span>
        <span style={{ color: fillColour }}>{pctStr}</span>
      </div>
      <div className="lg-edge-tooltip-bar">
        <div
          className="lg-edge-tooltip-bar-fill"
          style={{
            background: fillColour,
            transform: `scaleX(${Math.max(0.001, Math.min(1, pct))})`,
          }}
        />
      </div>
      <div className="lg-edge-tooltip-row">
        <span>used</span>
        <span>{used.toLocaleString()} tok</span>
      </div>
      <div className="lg-edge-tooltip-row">
        <span>limit</span>
        <span>{limit > 0 ? `${limit.toLocaleString()} tok` : '—'}</span>
      </div>
      {edge ? (
        <>
          <div className="lg-edge-tooltip-row">
            <span>kind</span>
            <span>{edge.kind}</span>
          </div>
          {edge.label ? (
            <div className="lg-edge-tooltip-row">
              <span>label</span>
              <code>{edge.label}</code>
            </div>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

export default function LiveSessionGraph(props: Props) {
  return (
    <ReactFlowProvider>
      <LiveGraphInner {...props} />
    </ReactFlowProvider>
  );
}
