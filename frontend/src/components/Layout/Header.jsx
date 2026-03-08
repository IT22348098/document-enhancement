import './Header.css';

function Header() {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-brand">
          <span className="header-logo">📄</span>
          <div>
            <h1 className="header-title">DocEnhance AI</h1>
            <p className="header-subtitle">Medical Report Image Enhancement</p>
          </div>
        </div>
        <div className="header-badge">
          <span className="badge-dot" />
          Powered by U-Net
        </div>
      </div>
    </header>
  );
}

export default Header;
