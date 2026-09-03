'use client';

import { useEffect, useMemo, useState } from 'react';

import {
  listMessagePrefixTemplates,
  removeMessagePrefixTemplate,
  saveMessagePrefixTemplate,
  type MessagePrefixTemplate,
  type Message,
  type ThinkingMode,
  type WindowState
} from '@/lib/api';

type Props = {
  sessionId: string | null;
  sessionTitle: string;
  sessionThinkingMode: ThinkingMode;
  sessionMessagePrefixPrompt: string;
  onUpdateSessionThinkingMode: (mode: ThinkingMode) => Promise<void>;
  onUpdateSessionMessagePrefixPrompt: (value: string) => Promise<void>;
  modelState: string;
  retrievalState: string;
  rolloverState: string;
  windowState: WindowState | null;
  sessionContextMode: 'full' | 'skill_state';
  // Real-time signals coming from the chat timeline — used for "what's
  // happening right now" without hitting the API again.
  lastEventAt: Date | null;
  messages: Message[];
  // Cumulative session totals — kept on the Session row by the backend.
  totalTokenCount: number;
  totalMessageCount: number;
};

function buildTemplateName(prompt: string): string {
  const normalized = prompt.trim().replace(/\s+/g, ' ');
  if (!normalized) {
    return '';
  }
  if (normalized.length <= 42) {
    return normalized;
  }
  return `${normalized.slice(0, 39)}...`;
}

// Compact humanization for token counts. 12_400 -> "12.4K", 1_200_000 -> "1.2M".
function fmtCompact(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return '0';
  if (n < 1_000) return String(n);
  if (n < 1_000_000) {
    const v = n / 1_000;
    return `${v >= 100 ? v.toFixed(0) : v.toFixed(1)}K`;
  }
  return `${(n / 1_000_000).toFixed(2)}M`;
}

// Context-bar fill colour: maps a 0..100 usage percent to a neutral ramp
// that mirrors the warm-amber accent. Low usage is muted grey, mid usage
// is the accent, and high usage is the danger red. No neon green/yellow.
function contextFillColor(percent: number, alpha = 0.95): string {
  const p = Math.max(0, Math.min(1, percent / 100));
  // Stops: 0.00 -> muted grey, 0.70 -> accent amber, 1.00 -> danger red.
  let r: number, g: number, b: number;
  if (p < 0.7) {
    const t = p / 0.7;
    r = Math.round(107 + (180 - 107) * t);
    g = Math.round(107 + (83 - 107) * t);
    b = Math.round(107 + (9 - 107) * t);
  } else {
    const t = (p - 0.7) / 0.3;
    r = Math.round(180 + (185 - 180) * t);
    g = Math.round(83 + (28 - 83) * t);
    b = Math.round(9 + (28 - 9) * t);
  }
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Maps a raw state string onto one of: idle | active | wait | error
// — we use this to drive the pill colour.
type PillTone = 'idle' | 'active' | 'wait' | 'error';
function toneForState(state: string): PillTone {
  if (!state || state === 'idle') return 'idle';
  if (state === 'error' || state === 'failed') return 'error';
  if (state === 'starting' || state === 'queued') return 'wait';
  return 'active'; // thinking / streaming / running / summarizing / background_run
}

// Human-friendly short label for the state — the panel is 180px wide
// so we keep the strings tight.
function shortLabel(state: string): string {
  if (!state) return 'idle';
  if (state === 'background_run') return 'bg run';
  if (state === 'streaming') return 'stream';
  if (state === 'summarizing') return 'summar.';
  return state;
}

// "Xs ago" — refreshes every second so the user always sees a fresh
// age without us re-fetching anything.
function formatAge(now: number, then: Date | null): string {
  if (!then) return '—';
  const diff = Math.max(0, Math.round((now - then.getTime()) / 1000));
  if (diff < 5) return 'now';
  if (diff < 60) return `${diff}s`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ${diff % 60}s`;
  const h = Math.floor(diff / 3600);
  const m = Math.floor((diff % 3600) / 60);
  return `${h}h ${m}m`;
}

// Recent activity tags extracted from the timeline. We map the chat
// message_type onto a tiny tag like the graph does, so the user can
// correlate at a glance.
const TYPE_TAG: Record<string, string> = {
  user: 'LLM',
  assistant: 'LLM',
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
  checkpoint: 'CK',
  window: 'WIN',
  auto_search_event: '🔎',
  rollover_summary: 'RO',
};

// Type → neutral color for the context progress bar and any other inline
// styling that needs a per-message-type hue. The palette is intentionally
// muted: the warm amber accent, the neutral text/muted greys, and a single
// danger red. No neon, no phosphor, no purple/blue/pink — those made the
// product feel like an AI demo instead of a tool.
const ACCENT = 'rgba(180, 83, 9, 0.95)';        // --accent
const ACCENT_SOFT = 'rgba(180, 83, 9, 0.55)';
const TEXT = 'rgba(26, 26, 26, 0.95)';          // --text
const MUTED = 'rgba(107, 107, 107, 0.95)';      // --muted
const MUTED_SOFT = 'rgba(107, 107, 107, 0.55)';
const WARNING = 'rgba(124, 90, 16, 0.95)';      // --warning-text
const WARNING_SOFT = 'rgba(124, 90, 16, 0.55)';
const DANGER = 'rgba(185, 28, 28, 0.95)';       // --danger
const DANGER_SOFT = 'rgba(185, 28, 28, 0.55)';

const TYPE_COLOR: Record<string, string> = {
  user: TEXT,
  assistant: ACCENT,
  tool_call: MUTED,
  tool_result: MUTED_SOFT,
  mcp_tool_call: WARNING,
  mcp_tool_result: WARNING_SOFT,
  terminal_command: WARNING,
  terminal_output: WARNING_SOFT,
  file_change: TEXT,
  file_snapshot: MUTED_SOFT,
  diff_summary: MUTED,
  test_result: ACCENT,
  build_error: DANGER,
  image_created: TEXT,
  artifact_created: MUTED,
  checkpoint: ACCENT_SOFT,
  window: ACCENT,
};

function truncate(str: string, n: number): string {
  if (!str) return '';
  return str.length <= n ? str : `${str.slice(0, n - 1)}…`;
}

// Pull a friendly tool name from the content_json payload when one is
// available; falls back to the raw type so the chip stays populated.
function toolHint(msg: Message): string {
  const cj = msg.content_json as Record<string, unknown> | undefined;
  const t = cj && typeof cj.tool === 'string' ? (cj.tool as string) : '';
  return t ? truncate(t, 10) : '';
}

export default function StatusPanel(props: Props) {
  const {
    sessionId,
    sessionTitle,
    sessionThinkingMode,
    sessionMessagePrefixPrompt,
    onUpdateSessionThinkingMode,
    onUpdateSessionMessagePrefixPrompt,
    modelState,
    retrievalState,
    rolloverState,
    windowState,
    sessionContextMode,
    lastEventAt,
    messages,
    totalTokenCount,
    totalMessageCount,
  } = props;

  // Live tick: forces a re-render every second so "Xs ago" updates
  // without us touching any data source.  The state is initialised
  // to ``null`` and only filled in client-side after mount, so the
  // server-rendered HTML always matches the first client render
  // (avoids the classic "Text content does not match server-rendered
  // HTML" hydration error caused by ``new Date()`` diverging between
  // SSR and the browser).
  const [now, setNow] = useState<number | null>(null);
  useEffect(() => {
    setNow(Date.now());
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, []);

  const percent = Math.round((windowState?.used_percent || 0) * 100);
  const used = windowState?.used_tokens ?? 0;
  const limit = windowState?.token_limit ?? 0;
  const free = Math.max(0, limit - used);
  const preAt = Math.round((windowState?.pre_rollover_threshold || 0) * 100);
  const hardAt = Math.round((windowState?.hard_rollover_threshold || 0) * 100);

  // Background-run count: any active SSE-delivered "background_run" model
  // state implies at least one. We further count distinct tool_call ids
  // whose matching tool_result hasn't been seen yet, as a low-cost
  // approximation of "in-flight" tool activity the user is waiting on.
  const activeRunCount = useMemo(() => {
    if (modelState === 'background_run') {
      // We can't enumerate runs without an API hit; one is the floor.
      const open = new Set<string>();
      const closed = new Set<string>();
      for (const m of messages) {
        if (m.message_type === 'tool_call') open.add(m.id);
        else if (m.message_type === 'tool_result') closed.add(m.id);
      }
      // background_run + at least one open tool call = a run doing real work
      for (const id of open) if (!closed.has(id)) return 2;
      return 1;
    }
    return 0;
  }, [modelState, messages]);

  const recentActivity = useMemo(() => {
    const tail = messages.slice(-8);
    // Walk back-to-front so the *latest* event is the leftmost chip.
    return tail.slice().reverse().map((m) => {
      const tag = TYPE_TAG[m.message_type] ?? m.message_type.slice(0, 3).toUpperCase();
      const colour = TYPE_COLOR[m.message_type] ?? 'rgba(180, 83, 9, 0.85)';
      const hint = toolHint(m);
      return { id: m.id, tag, colour, hint, role: m.role };
    });
  }, [messages]);

  // Win = how many "window" boundary nodes we've crossed (i.e. how
  // many times the orchestrator has rolled over). Cheap to derive.
  const windowCount = useMemo(() => {
    return messages.filter((m) => m.message_type === 'window' || m.message_type === 'checkpoint').length;
  }, [messages]);

  // Pill data: keep the layout column-stable (always 2 columns) so the
  // dashboard doesn't reflow when one state toggles.
  const pills = [
    { key: 'model', label: 'Model', state: modelState, tone: toneForState(modelState), short: shortLabel(modelState) },
    { key: 'retr', label: 'Retrieval', state: retrievalState, tone: toneForState(retrievalState), short: shortLabel(retrievalState) },
    { key: 'roll', label: 'Rollover', state: rolloverState, tone: toneForState(rolloverState), short: shortLabel(rolloverState) },
    {
      key: 'runs',
      label: 'Runs',
      state: activeRunCount > 0 ? 'running' : 'idle',
      tone: activeRunCount > 0 ? 'active' : 'idle',
      short: activeRunCount > 0 ? `${activeRunCount} active` : 'idle',
    },
  ];

  // Live pulse: the top dot pulses green if the model is active in
  // the last ~3s, dim idle otherwise. We piggy-back on the lastEventAt
  // age — if we got a chat event very recently, we treat the system
  // as "live".  ``now`` is ``null`` until the first client tick, so we
  // fall back to a stable placeholder on SSR / first paint to avoid
  // hydration mismatch (server clock ≠ client clock).
  const isLive = !!lastEventAt && now !== null && now - lastEventAt.getTime() < 3000;
  const age = now === null ? '—' : formatAge(now, lastEventAt);
  const nowStr = useMemo(() => {
    if (now === null) return '--:--:--';
    const d = new Date(now);
    const hh = String(d.getHours()).padStart(2, '0');
    const mm = String(d.getMinutes()).padStart(2, '0');
    const ss = String(d.getSeconds()).padStart(2, '0');
    return `${hh}:${mm}:${ss}`;
  }, [now]);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [messagePrefixDraft, setMessagePrefixDraft] = useState(sessionMessagePrefixPrompt || '');
  const [savingMessagePrefix, setSavingMessagePrefix] = useState(false);
  const [savingThinkingMode, setSavingThinkingMode] = useState(false);
  const [messagePrefixTemplates, setMessagePrefixTemplates] = useState<MessagePrefixTemplate[]>([]);
  const [selectedTemplateId, setSelectedTemplateId] = useState('');
  const [loadingTemplates, setLoadingTemplates] = useState(false);
  const [updatingTemplateList, setUpdatingTemplateList] = useState(false);

  useEffect(() => {
    setMessagePrefixDraft(sessionMessagePrefixPrompt || '');
  }, [sessionId, sessionMessagePrefixPrompt]);

  useEffect(() => {
    void reloadTemplates();
  }, []);

  async function reloadTemplates(preferredTemplateId?: string) {
    setLoadingTemplates(true);
    try {
      const loaded = await listMessagePrefixTemplates();
      setMessagePrefixTemplates(loaded);
      setSelectedTemplateId((prev) => {
        const candidate = preferredTemplateId || prev;
        if (candidate && loaded.some((item) => item.id === candidate)) {
          return candidate;
        }
        return loaded[0]?.id || '';
      });
    } catch {
      setMessagePrefixTemplates([]);
      setSelectedTemplateId('');
    } finally {
      setLoadingTemplates(false);
    }
  }

  async function onApplyMessagePrefixPrompt(overrideValue?: string) {
    if (!sessionId || savingMessagePrefix) {
      return;
    }
    const nextPrompt = overrideValue ?? messagePrefixDraft;
    if (nextPrompt === (sessionMessagePrefixPrompt || '')) {
      return;
    }
    setSavingMessagePrefix(true);
    try {
      await onUpdateSessionMessagePrefixPrompt(nextPrompt);
    } finally {
      setSavingMessagePrefix(false);
    }
  }

  async function onUseTemplate() {
    if (!sessionId || !selectedTemplateId || loadingTemplates || updatingTemplateList) {
      return;
    }
    const template = messagePrefixTemplates.find((item) => item.id === selectedTemplateId);
    if (!template) {
      return;
    }
    setMessagePrefixDraft(template.prompt);
    await onApplyMessagePrefixPrompt(template.prompt);
  }

  async function onSaveTemplate() {
    const prompt = messagePrefixDraft.trim();
    if (!prompt || loadingTemplates || updatingTemplateList) {
      return;
    }
    const currentTemplate = messagePrefixTemplates.find((item) => item.id === selectedTemplateId) || null;
    const defaultName =
      currentTemplate && currentTemplate.prompt === prompt ? currentTemplate.name : buildTemplateName(prompt);
    const name = window.prompt('Template name', defaultName)?.trim() || '';
    if (!name) {
      return;
    }

    setUpdatingTemplateList(true);
    try {
      const saved = await saveMessagePrefixTemplate({ name, prompt });
      await reloadTemplates(saved.id);
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'Failed to save template';
      window.alert(detail);
    } finally {
      setUpdatingTemplateList(false);
    }
  }

  async function onDeleteTemplate() {
    if (!selectedTemplateId || loadingTemplates || updatingTemplateList) {
      return;
    }
    const currentTemplate = messagePrefixTemplates.find((item) => item.id === selectedTemplateId);
    if (!currentTemplate) {
      return;
    }
    const ok = window.confirm(`Delete template "${currentTemplate.name}"?`);
    if (!ok) {
      return;
    }

    setUpdatingTemplateList(true);
    try {
      await removeMessagePrefixTemplate(currentTemplate.id);
      await reloadTemplates();
    } catch (err) {
      const detail = err instanceof Error ? err.message : 'Failed to delete template';
      window.alert(detail);
    } finally {
      setUpdatingTemplateList(false);
    }
  }

  const isTemplateBusy = loadingTemplates || updatingTemplateList;

  // Threshold ticks are positioned in % along the bar; we render them
  // as absolutely-positioned spans so they overlay the <progress>.
  const prePct = Math.max(0, Math.min(100, preAt));
  const hardPct = Math.max(0, Math.min(100, hardAt));
  const fillColour = contextFillColor(windowState?.used_percent ?? 0, 0.95);

  return (
    <section className="panel status-panel" style={{ position: 'relative' }}>
      <button
        type="button"
        className="status-panel-open-btn"
        onClick={() => setSettingsOpen(true)}
        title="Session settings"
      >
        ⚙ Settings
      </button>
      <h2>Status</h2>

      {/* Live header: pulse + age + clock */}
      <div className="status-live-row">
        <span className={`status-live-dot ${isLive ? 'live' : 'idle'}`} aria-hidden="true" />
        <span className="status-live-label">
          {isLive ? 'live' : 'idle'} · {age}
        </span>
        <span className="status-live-clock">{nowStr}</span>
      </div>

      {/* Session title — single line, truncates, full title in tooltip */}
      <div
        className="status-session-title"
        title={sessionTitle || 'no session'}
      >
        {sessionTitle ? truncate(sessionTitle, 28) : <span className="status-muted">no session</span>}
      </div>

      {/* State pills grid — 2 cols, stable layout */}
      <div className="status-pills">
        {pills.map((p) => (
          <div key={p.key} className={`status-pill status-pill-${p.tone}`} title={`${p.label}: ${p.state}`}>
            <span className="status-pill-dot" />
            <span className="status-pill-label">{p.label}</span>
            <span className="status-pill-state">{p.short}</span>
          </div>
        ))}
      </div>

      {/* Context meter with threshold ticks */}
      <div className="status-context">
        <div className="status-context-head">
          <span className="status-context-title">CONTEXT</span>
          <span className="status-context-main">
            {fmtCompact(used)}<span className="status-muted">/{fmtCompact(limit)}</span>
            <span className="status-context-pct">{percent}%</span>
          </span>
        </div>
        <div className="status-context-bar">
          <progress max={100} value={percent} style={{ accentColor: fillColour }} />
          {preAt > 0 ? (
            <span
              className="status-context-tick status-context-tick-pre"
              style={{ left: `${prePct}%` }}
              title={`Pre-rollover @ ${prePct}%`}
            />
          ) : null}
          {hardAt > 0 ? (
            <span
              className="status-context-tick status-context-tick-hard"
              style={{ left: `${hardPct}%` }}
              title={`Hard rollover @ ${hardPct}%`}
            />
          ) : null}
        </div>
        <div className="status-context-meta">
          <span>free {fmtCompact(free)}</span>
          {preAt > 0 ? <span>pre @{prePct}%</span> : null}
          {hardAt > 0 ? <span>hard @{hardPct}%</span> : null}
        </div>
      </div>

      {/* Session mode + counters */}
      <div className="status-mode-row">
        <span
          className={`status-mode-pill status-mode-${sessionContextMode}`}
          title="Locked at session creation. Create a new session to switch."
        >
          {sessionContextMode === 'skill_state' ? 'SKILL.state' : 'Full session'}
        </span>
      </div>

      <div className="status-stats">
        <span title="Cumulative session tokens">
          <span className="status-muted">Σ</span> {fmtCompact(totalTokenCount)}
        </span>
        <span title="Messages in this session">msgs {totalMessageCount}</span>
        <span title="Windows / checkpoints seen">win {windowCount}</span>
      </div>

      {/* Recent activity stream — leftmost = most recent */}
      <div className="status-activity">
        <div className="status-activity-head">
          <span>RECENT</span>
          <span className="status-muted">{recentActivity.length > 0 ? `${recentActivity.length}/8` : '—'}</span>
        </div>
        {recentActivity.length > 0 ? (
          <div className="status-activity-chips">
            {recentActivity.map((a) => (
              <span
                key={a.id}
                className="status-activity-chip"
                style={{ borderColor: a.colour, color: a.colour }}
                title={`${a.role}: ${a.tag}${a.hint ? ` (${a.hint})` : ''}`}
              >
                {a.tag}
                {a.hint ? <span className="status-activity-hint">{a.hint}</span> : null}
              </span>
            ))}
          </div>
        ) : (
          <div className="status-activity-empty">no events yet</div>
        )}
      </div>

      {settingsOpen ? (
        <div className="status-modal-backdrop" onClick={() => setSettingsOpen(false)}>
          <div className="status-modal" onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h2 style={{ margin: 0 }}>Session Settings</h2>
              <button onClick={() => setSettingsOpen(false)}>×</button>
            </div>
            <div className="status-settings" style={{ display: 'block', marginTop: 12 }}>
              <label className="status-field" style={{ display: 'block', marginBottom: 12 }}>
                <span style={{ display: 'block', marginBottom: 4 }}>Session thinking</span>
                <select
                  value={sessionThinkingMode}
                  disabled={!sessionId || savingThinkingMode}
                  onChange={async (e) => {
                    if (!sessionId) {
                      return;
                    }
                    const mode = e.target.value as ThinkingMode;
                    if (mode === sessionThinkingMode) {
                      return;
                    }
                    setSavingThinkingMode(true);
                    try {
                      await onUpdateSessionThinkingMode(mode);
                    } finally {
                      setSavingThinkingMode(false);
                    }
                  }}
                >
                  <option value="off">off</option>
                  <option value="low">low</option>
                  <option value="medium">medium</option>
                  <option value="high">high</option>
                </select>
              </label>

              <label className="status-field" style={{ display: 'block', marginBottom: 12 }}>
                <span style={{ display: 'block', marginBottom: 4 }}>Per-message prompt</span>
                <div className="status-inline-row" style={{ display: 'flex', gap: 6 }}>
                  <input
                    value={messagePrefixDraft}
                    onChange={(e) => setMessagePrefixDraft(e.target.value)}
                    onBlur={() => {
                      void onApplyMessagePrefixPrompt();
                    }}
                    placeholder="prefix for every user turn"
                    style={{ flex: 1 }}
                    disabled={!sessionId || savingMessagePrefix}
                  />
                  <button
                    type="button"
                    onClick={() => {
                      void onApplyMessagePrefixPrompt();
                    }}
                    disabled={!sessionId || savingMessagePrefix || messagePrefixDraft === (sessionMessagePrefixPrompt || '')}
                  >
                    {savingMessagePrefix ? 'Saving...' : 'Apply'}
                  </button>
                </div>
              </label>

              <div className="status-template-controls">
                <select
                  value={selectedTemplateId}
                  onChange={(e) => setSelectedTemplateId(e.target.value)}
                  disabled={isTemplateBusy}
                  style={{ marginRight: 6 }}
                >
                  <option value="">
                    {loadingTemplates
                      ? 'Loading templates...'
                      : messagePrefixTemplates.length > 0
                        ? 'Select template'
                        : 'No templates yet'}
                  </option>
                  {messagePrefixTemplates.map((template) => (
                    <option key={template.id} value={template.id}>
                      {template.name}
                    </option>
                  ))}
                </select>
                <div className="status-template-actions" style={{ display: 'inline-flex', gap: 6 }}>
                  <button
                    type="button"
                    onClick={() => void onUseTemplate()}
                    disabled={!sessionId || !selectedTemplateId || isTemplateBusy}
                  >
                    Use
                  </button>
                  <button type="button" onClick={() => void onSaveTemplate()} disabled={!messagePrefixDraft.trim() || isTemplateBusy}>
                    {updatingTemplateList ? 'Saving...' : 'Save'}
                  </button>
                  <button type="button" onClick={() => void onDeleteTemplate()} disabled={!selectedTemplateId || isTemplateBusy}>
                    Delete
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}