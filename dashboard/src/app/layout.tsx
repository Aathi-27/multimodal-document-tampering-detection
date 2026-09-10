import type { Metadata } from 'next';
import '@/styles/globals.css';

export const metadata: Metadata = {
  title: 'Document Tampering Detection — Dashboard',
  description: 'Real-time document fraud detection dashboard with 6-signal CV+OCR fusion pipeline',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <div className="flex">
          {/* Sidebar */}
          <aside className="w-64 min-h-screen bg-slate-900 border-r border-slate-700 p-4 fixed">
            <div className="mb-8">
              <h1 className="text-xl font-bold text-blue-400">🔍 DocTamper</h1>
              <p className="text-xs text-slate-400 mt-1">v2.0 — 6-Signal Fusion</p>
            </div>
            <nav className="space-y-2">
              <a href="/" className="flex items-center gap-3 px-3 py-2 rounded-lg bg-slate-800 text-white">
                <span>📊</span> Dashboard
              </a>
              <a href="/analyze" className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-slate-800 text-slate-300 transition">
                <span>🔎</span> Analyze
              </a>
              <a href="/history" className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-slate-800 text-slate-300 transition">
                <span>📜</span> History
              </a>
              <a href="/settings" className="flex items-center gap-3 px-3 py-2 rounded-lg hover:bg-slate-800 text-slate-300 transition">
                <span>⚙️</span> Settings
              </a>
            </nav>
            <div className="absolute bottom-4 left-4 right-4">
              <div className="card text-xs text-slate-400">
                <p className="font-semibold text-slate-300 mb-1">Pipeline Signals</p>
                <ul className="space-y-0.5">
                  <li>• ELA Analysis</li>
                  <li>• Grad-CAM</li>
                  <li>• MC Dropout</li>
                  <li>• OCR Validation</li>
                  <li>• Patch Localization</li>
                  <li>• Spatial Overlap</li>
                </ul>
              </div>
            </div>
          </aside>

          {/* Main Content */}
          <main className="flex-1 ml-64 p-8">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
