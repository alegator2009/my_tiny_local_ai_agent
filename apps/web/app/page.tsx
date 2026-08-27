'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';

import ChatPanel from '@/components/ChatPanel';
import EvolutionPanel from '@/components/EvolutionPanel';
import LiveSessionGraph from '@/components/LiveSessionGraph';
import SessionList from '@/components/SessionList';
import StatusPanel from '@/components/StatusPanel';
import {
  archiveSession,
  createSession,
  getSessionGraph,
  getTranscript,
  getWindowState,
  listSessions,
  removeSession,
  updateSession,
  type Message,
  type Session,
  type ThinkingMode,
  type WindowState
} from '@/lib/api';

export default function HomePage() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [windowState, setWindowState] = useState<WindowState | null>(null);
  const [modelState, setModelState] = useState('idle');
  const [retrievalState, setRetrievalState] = useState('idle');
  const [rolloverState, setRolloverState] = useState('idle');

  async function reloadSessions() {
    const data = await listSessions();
    setSessions(data);
    if (!selectedSessionId && data.length > 0) {
      setSelectedSessionId(data[0].id);
    }
  }

  useEffect(() => {
    void reloadSessions();
  }, []);

  useEffect(() => {
    if (!selectedSessionId) {
      setMessages([]);
      setWindowState(null);
      return;
    }
    void getTranscript(selectedSessionId).then(setMessages);
    void getWindowState(selectedSessionId).then(setWindowState);

    // Background poll: refresh transcript + window state every 3s while a
    // session is open. Without this, messages added by an external caller
    // (CLI, tests, REST API) only appear after a manual page refresh.
    // We pause the poll while the chat panel is actively streaming so we
    // don't fight the SSE deltas.
    let cancelled = false;
    let lastMessageCount = -1;
    const tick = () => {
      if (cancelled) return;
      // Skip if the user is currently streaming.
      if (modelState === 'thinking' || modelState === 'streaming') {
        return;
      }
      void getTranscript(selectedSessionId)
        .then((transcript) => {
          if (cancelled) return;
          if (transcript.length !== lastMessageCount) {
            lastMessageCount = transcript.length;
            setMessages(transcript);
          }
        })
        .catch(() => undefined);
      void getWindowState(selectedSessionId)
        .then((ws) => {
          if (!cancelled && ws) setWindowState(ws);
        })
        .catch(() => undefined);
    };
    // Seed the count so we don't trigger an extra render on the first tick.
    void getTranscript(selectedSessionId).then((t) => {
      if (cancelled) return;
      lastMessageCount = t.length;
    });
    const timer = window.setInterval(tick, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [selectedSessionId, modelState]);

  const selectedSession = useMemo(
    () => sessions.find((s) => s.id === selectedSessionId) || null,
    [sessions, selectedSessionId]
  );

  async function updateMessagePrefixPromptForSession(value: string) {
    if (!selectedSession) {
      return;
    }
    const updated = await updateSession(selectedSession.id, { message_prefix_prompt: value });
    setSessions((prev) => prev.map((session) => (session.id === updated.id ? updated : session)));
  }

  async function updateThinkingModeForSession(mode: ThinkingMode) {
    if (!selectedSession) {
      return;
    }
    const updated = await updateSession(selectedSession.id, { thinking_mode: mode });
    setSessions((prev) => prev.map((session) => (session.id === updated.id ? updated : session)));
  }

  // Track which tool_call ids are awaiting their result. Derived from
  // the current message list: a tool_call is "in-flight" until we see a
  // matching tool_result that follows it.
  const inFlight = useMemo<Set<string>>(() => {
    const out = new Set<string>();
    const open = new Set<string>();
    for (const m of messages) {
      if (m.message_type === 'tool_call') {
        open.add(m.id);
      } else if (m.message_type === 'tool_result') {
        // naive 1:1 FIFO match — same approach as the backend graph builder
        const it = open.values().next().value;
        if (it) {
          open.delete(it);
        }
      }
    }
    for (const id of open) {
      out.add(`msg:${id}`);
    }
    return out;
  }, [messages]);

  // Live cognitive-state hint: the most recent SSE event that signals
  // activity. We pass it down to LiveSessionGraph as a category tag so
  // the corresponding "cognitive area" (LLM / MCP / memory / tools)
  // can pulse even before the corresponding graph node arrives via
  // refetch. Keys are synthetic ids recognised by the renderer.
  const [liveHint, setLiveHint] = useState<{ kind: string; ts: number } | null>(null);
  useEffect(() => {
    // model_status "thinking"/"streaming" -> LLM area
    if (modelState === 'thinking' || modelState === 'streaming' || modelState === 'background_run') {
      setLiveHint({ kind: 'llm', ts: Date.now() });
      return;
    }
    if (retrievalState === 'running' || retrievalState === 'starting') {
      setLiveHint({ kind: 'memory', ts: Date.now() });
      return;
    }
    if (rolloverState === 'summarizing' || rolloverState === 'starting') {
      setLiveHint({ kind: 'checkpoint', ts: Date.now() });
    }
  }, [modelState, retrievalState, rolloverState]);
  const liveInFlight = useMemo<Set<string>>(() => {
    if (!liveHint || Date.now() - liveHint.ts > 8000) return inFlight;
    // The renderer maps these synthetic ids to the right visual
    // depending on what's in the current graph.
    const out = new Set(inFlight);
    out.add(`__live__:${liveHint.kind}`);
    return out;
  }, [inFlight, liveHint]);

  // The graph component handles its own incremental refresh via lastEventAt,
  // so we don't need to remount it on every new message.

  // Also refresh window state
  useEffect(() => {
    if (!selectedSessionId) {
      return;
    }
    const t = setInterval(() => {
      void getWindowState(selectedSessionId).then(setWindowState).catch(() => {});
    }, 4000);
    return () => clearInterval(t);
  }, [selectedSessionId]);

  // Graph size — track container width
  const [graphSize, setGraphSize] = useState<{ w: number; h: number }>({ w: 360, h: 600 });
  const mainColRef = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    const update = () => {
      // graph is the 4th column at ~380px; we measure via window for now
      setGraphSize({ w: Math.max(320, Math.min(420, window.innerWidth - 700)), h: window.innerHeight - 80 });
    };
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  return (
    <main className="layout">
      <SessionList
        sessions={sessions}
        selectedSessionId={selectedSessionId}
        onSelect={setSelectedSessionId}
        onCreate={async ({ title, context_mode }) => {
          // The modal owns the context_mode picker — that's the ONE place
          // the user can ever set it. After creation it is locked.
          // We still carry over the chat-header toggles from the active
          // session (or the global defaults) so a new chat starts in a
          // useful state.
          const ref = sessions.find((s) => s.id === selectedSessionId);
          const created = await createSession({
            title,
            context_mode,
            thinking_mode:
              (ref?.thinking_mode as 'off' | 'low' | 'medium' | 'high' | undefined) ??
              'medium',
            force_search_next: ref?.force_search_next ?? false,
            bypass_search_cache_next: ref?.bypass_search_cache_next ?? false,
            provider_id: ref?.provider_id ?? null,
            model_id: ref?.model_id ?? null
          });
          await reloadSessions();
          setSelectedSessionId(created.id);
        }}
        onArchive={async (id) => {
          await archiveSession(id);
          await reloadSessions();
        }}
        onDelete={async (id) => {
          await removeSession(id);
          if (selectedSessionId === id) {
            setSelectedSessionId(null);
          }
          await reloadSessions();
        }}
      />

      <section className="main-column" ref={mainColRef}>
        <header className="toolbar">
          <div className="toolbar-main">
            <h1>{selectedSession?.title || 'Select a session'}</h1>
          </div>
          <nav>
            <Link href="/settings">Settings</Link>
            {selectedSessionId ? <Link href={`/memory/${selectedSessionId}`}>Memory</Link> : null}
          </nav>
        </header>

        <EvolutionPanel />

        <ChatPanel
          sessionId={selectedSessionId}
          messages={messages}
          onMessages={setMessages}
          onModelState={setModelState}
          onRetrievalState={setRetrievalState}
          onRolloverState={setRolloverState}
          onWindowState={setWindowState}
          sessionThinkingMode={selectedSession?.thinking_mode || 'medium'}
          sessionProviderId={selectedSession?.provider_id ?? null}
          sessionModelId={selectedSession?.model_id ?? null}
          sessionHideSystemMessages={selectedSession?.hide_system_messages ?? false}
          sessionRunInBackground={selectedSession?.run_in_background ?? false}
          sessionForceSearchNext={selectedSession?.force_search_next ?? false}
          sessionBypassSearchCacheNext={selectedSession?.bypass_search_cache_next ?? false}
          sessionContextMode={selectedSession?.context_mode ?? 'full'}
          onSessionUpdated={(patch) => {
            setSessions((prev) =>
              prev.map((s) =>
                s.id === selectedSessionId
                  ? {
                      ...s,
                      provider_id:
                        patch.provider_id !== undefined
                          ? patch.provider_id
                          : s.provider_id,
                      model_id:
                        patch.model_id !== undefined ? patch.model_id : s.model_id,
                      thinking_mode:
                        patch.thinking_mode !== undefined
                          ? patch.thinking_mode
                          : s.thinking_mode,
                      hide_system_messages:
                        patch.hide_system_messages !== undefined
                          ? patch.hide_system_messages
                          : s.hide_system_messages,
                      run_in_background:
                        patch.run_in_background !== undefined
                          ? patch.run_in_background
                          : s.run_in_background,
                      force_search_next:
                        patch.force_search_next !== undefined
                          ? patch.force_search_next
                          : s.force_search_next,
                      bypass_search_cache_next:
                        patch.bypass_search_cache_next !== undefined
                          ? patch.bypass_search_cache_next
                          : s.bypass_search_cache_next
                      // ``context_mode`` is intentionally NOT mutable here.
                      // The runtime is locked at session creation; only the
                      // new-session modal can ever set it.
                    }
                  : s
              )
            );
          }}
        />
      </section>

      <StatusPanel
        sessionId={selectedSessionId}
        sessionTitle={selectedSession?.title || ''}
        sessionThinkingMode={selectedSession?.thinking_mode || 'medium'}
        sessionMessagePrefixPrompt={selectedSession?.message_prefix_prompt || ''}
        onUpdateSessionThinkingMode={updateThinkingModeForSession}
        onUpdateSessionMessagePrefixPrompt={updateMessagePrefixPromptForSession}
        modelState={modelState}
        retrievalState={retrievalState}
        rolloverState={rolloverState}
        windowState={windowState}
        sessionContextMode={selectedSession?.context_mode ?? 'full'}
        lastEventAt={messages.length > 0 ? new Date(messages[messages.length - 1].timestamp) : null}
        messages={messages}
        totalTokenCount={selectedSession?.total_token_count ?? 0}
        totalMessageCount={selectedSession?.total_message_count ?? messages.length}
      />

      <aside className="panel graph-panel" style={{ padding: 8, position: 'relative' }}>
        <LiveSessionGraph
          sessionId={selectedSessionId}
          width={graphSize.w}
          height={graphSize.h}
          inFlight={liveInFlight}
          lastEventAt={messages.length > 0 ? new Date(messages[messages.length - 1].timestamp) : null}
          windowState={windowState}
        />
      </aside>
    </main>
  );
}
