'use client';

import Link from 'next/link';
import { useState } from 'react';

import type { Session } from '@/lib/api';

type Props = {
  sessions: Session[];
  selectedSessionId: string | null;
  onSelect: (id: string) => void;
  // The create modal returns both the title and the context mode the user
  // picked at creation time. The context mode is locked for the lifetime
  // of the session — the caller (page.tsx) is expected to send it to the
  // backend and never change it again.
  onCreate: (input: { title: string; context_mode: 'full' | 'skill_state' }) => Promise<void>;
  onArchive: (id: string) => Promise<void>;
  onDelete: (id: string) => Promise<void>;
};

export default function SessionList({ sessions, selectedSessionId, onSelect, onCreate, onArchive, onDelete }: Props) {
  const [createOpen, setCreateOpen] = useState(false);
  const [titleDraft, setTitleDraft] = useState('');
  const [modeDraft, setModeDraft] = useState<'full' | 'skill_state'>('full');
  const [submitting, setSubmitting] = useState(false);

  const reset = () => {
    setTitleDraft('');
    setModeDraft('full');
  };

  return (
    <aside className="panel session-panel">
      <div className="panel-head">
        <h2>Sessions</h2>
        <button
          onClick={() => {
            reset();
            setCreateOpen(true);
          }}
        >
          New
        </button>
      </div>
      <ul className="session-list">
        {sessions.map((s) => (
          <li key={s.id} className={selectedSessionId === s.id ? 'active' : ''}>
            <button className="session-row" onClick={() => onSelect(s.id)}>
              <span>{s.title}</span>
              <small>{new Date(s.updated_at).toLocaleString()}</small>
            </button>
            <div className="session-actions">
              <button onClick={() => onArchive(s.id)}>Archive</button>
              <button
                onClick={async () => {
                  const ok = window.confirm(`Delete session "${s.title}" permanently?`);
                  if (!ok) {
                    return;
                  }
                  await onDelete(s.id);
                }}
              >
                Delete
              </button>
            </div>
          </li>
        ))}
      </ul>

      {createOpen ? (
        <div
          className="status-modal-backdrop"
          role="dialog"
          aria-modal="true"
          aria-label="New session"
          onClick={() => {
            if (!submitting) setCreateOpen(false);
          }}
        >
          <div className="status-modal" onClick={(e) => e.stopPropagation()}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}
            >
              <h2 style={{ margin: 0 }}>New session</h2>
              <button
                onClick={() => {
                  if (!submitting) setCreateOpen(false);
                }}
                disabled={submitting}
                aria-label="Close"
              >
                ×
              </button>
            </div>
            <p
              className="small-muted"
              style={{ marginTop: 8, marginBottom: 12 }}
            >
              Pick the runtime for this session. Once created, the choice is
              locked for the lifetime of the session — it can&apos;t be
              changed later.
            </p>

            <label className="status-field" style={{ display: 'block', marginBottom: 12 }}>
              <span style={{ display: 'block', marginBottom: 4 }}>Title</span>
              <input
                type="text"
                value={titleDraft}
                onChange={(e) => setTitleDraft(e.target.value)}
                placeholder="e.g. fusion-news-research"
                style={{ width: '100%', boxSizing: 'border-box' }}
                disabled={submitting}
                autoFocus
              />
            </label>

            <fieldset
              className="status-field"
              style={{
                display: 'block',
                border: '1px solid var(--panel-border, #444)',
                borderRadius: 6,
                padding: 8,
                marginBottom: 12
              }}
              disabled={submitting}
            >
              <legend style={{ padding: '0 4px' }}>Context mode</legend>
              <label
                style={{
                  display: 'flex',
                  gap: 8,
                  alignItems: 'flex-start',
                  marginBottom: 8,
                  cursor: 'pointer'
                }}
              >
                <input
                  type="radio"
                  name="context-mode"
                  value="full"
                  checked={modeDraft === 'full'}
                  onChange={() => setModeDraft('full')}
                  style={{ marginTop: 4 }}
                />
                <span>
                  <strong>Full session</strong>
                  <br />
                  <small className="small-muted">
                    Replay the whole chat history to the model on every turn.
                    Default behaviour.
                  </small>
                </span>
              </label>
              <label
                style={{
                  display: 'flex',
                  gap: 8,
                  alignItems: 'flex-start',
                  cursor: 'pointer'
                }}
              >
                <input
                  type="radio"
                  name="context-mode"
                  value="skill_state"
                  checked={modeDraft === 'skill_state'}
                  onChange={() => setModeDraft('skill_state')}
                  style={{ marginTop: 4 }}
                />
                <span>
                  <strong>SKILL.state</strong>
                  <br />
                  <small className="small-muted">
                    Only (spec, structured state, latest observation) goes to
                    the model. Reasoning traces are discarded after each step.
                    Ideal for long-running, structured procedures.
                  </small>
                </span>
              </label>
            </fieldset>

            <div
              style={{
                display: 'flex',
                gap: 8,
                justifyContent: 'flex-end',
                marginTop: 16
              }}
            >
              <button
                onClick={() => {
                  if (!submitting) setCreateOpen(false);
                }}
                disabled={submitting}
              >
                Cancel
              </button>
              <button
                onClick={async () => {
                  const title = titleDraft.trim();
                  if (!title) return;
                  setSubmitting(true);
                  try {
                    await onCreate({ title, context_mode: modeDraft });
                    setCreateOpen(false);
                    reset();
                  } finally {
                    setSubmitting(false);
                  }
                }}
                disabled={!titleDraft.trim() || submitting}
                className="primary"
              >
                {submitting ? 'Creating…' : 'Create'}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </aside>
  );
}