import ThemeToggle from "./ThemeToggle";

function Header({ theme, toggleTheme }) {
  return (
    <div className="header">
      <ThemeToggle theme={theme} toggleTheme={toggleTheme} />
      <div className="logo">
        <div className="logo-icon">
          <div className="logo-sparkle"></div>
        </div>
      </div>
      <h1>NIMA</h1>
      <p className="subtitle">Intelligent Text Generation</p>
    </div>
  );
}

export default Header;
