declare module 'dagre' {
  export interface GraphLabel {
    width?: number;
    height?: number;
    rankdir?: 'TB' | 'BT' | 'LR' | 'RL';
    ranksep?: number;
    nodesep?: number;
    edgesep?: number;
    marginx?: number;
    marginy?: number;
  }

  export interface NodeLabel {
    width?: number;
    height?: number;
    label?: string;
    [key: string]: unknown;
  }

  export interface EdgeLabel {
    width?: number;
    height?: number;
    label?: string;
    minlen?: number;
    weight?: number;
    [key: string]: unknown;
  }

  export class Graph {
    constructor(opt?: GraphLabel);
    setDefaultEdgeLabel(callback: () => EdgeLabel): void;
    setGraph(label: GraphLabel): void;
    setNode(id: string, label?: NodeLabel): void;
    setEdge(source: string, target: string, label?: EdgeLabel): void;
    node(id: string): NodeLabel | undefined;
    node(id: string, label: NodeLabel): void;
    edge(source: string, target: string, edgeObj?: EdgeLabel): EdgeLabel | undefined;
    nodes(): { [id: string]: NodeLabel };
    edges(): Array<{ v: string; w: string; name?: string }>;
    removeNode(id: string): void;
    removeEdge(source: string, target: string): void;
  }

  export function graphlib(): { Graph: typeof Graph };
  export function layout(graph: Graph): void;
  const _default: { graphlib: { Graph: typeof Graph }; layout: typeof layout };
  export default _default;
}
