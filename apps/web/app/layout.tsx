import './globals.css';
import type { ReactNode } from 'react';

import ThemeToggle from '@/components/ThemeToggle';

export const metadata = {
  title: 'AI Infinite Session',
  description: 'Local-first infinite agent sessions with rollover and retrieval'
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>
        <ThemeToggle />
        {children}
      </body>
    </html>
  );
}
