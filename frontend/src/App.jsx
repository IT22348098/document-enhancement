import { useState, useCallback } from 'react';
import Header from './components/Header.jsx';
import EnhancementFeature from './features/enhancement/index.jsx';
import './App.css';

function App() {
  const [toast, setToast] = useState(null); // { message, type: 'success'|'error' }

  const showToast = useCallback((message, type = 'success') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 3500);
  }, []);

  return (
    <div className="app">
      <Header />

      <main className="app-main">
        {/* ── Image Enhancement feature (IT22348098) ── */}
        <EnhancementFeature onToast={showToast} />

        {/* ── Add other teammates' features below ─────
        <AnotherFeature />
        ──────────────────────────────────────────── */}
      </main>

      {toast && (
        <div className={`toast toast--${toast.type}`}>
          {toast.message}
        </div>
      )}
    </div>
  );
}

export default App;

