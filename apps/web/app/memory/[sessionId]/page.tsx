'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { useParams } from 'next/navigation';

import { getMemorySnapshot, runMemoryLint } from '@/lib/api';

export default function MemoryPage() {
  const params = useParams<{ sessionId: string }>();
  const sessionId = params?.sessionId;
  const [snapshot, setSnapshot] = useState<any>(null);
  const [linting, setLinting] = useState(false);

  async function reload() {
    if (!sessionId) {
      return;
    }
    const data = await getMemorySnapshot(sessionId);
    setSnapshot(data);
  }

  useEffect(() => {
    void reload();
  }, [sessionId]);

  return (
    <main className="simple-page">
      <header className="page-head">
        <div>
          <h1>Memory Inspector</h1>
        </div>
        <Link href="/" className="page-head-action">Back</Link>
      </header>

      <div style={{ display: 'flex', gap: 8 }}>
        <button
          onClick={async () => {
            if (!sessionId || linting) {
              return;
            }
            setLinting(true);
            try {
              await runMemoryLint(sessionId, 'manual_ui');
              await reload();
            } finally {
              setLinting(false);
            }
          }}
        >
          {linting ? 'Running lint...' : 'Run wiki lint'}
        </button>
        <button onClick={() => void reload()}>Refresh</button>
      </div>

      {!snapshot ? <p>Loading...</p> : <pre>{JSON.stringify(snapshot, null, 2)}</pre>}
    </main>
  );
}
