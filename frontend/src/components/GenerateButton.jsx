function GenerateButton({ onClick, isLoading, theme }) {
  return (
    <button
      className={`generate-btn ${theme}`}
      onClick={onClick}
      disabled={isLoading}
    >
      {isLoading ? (
        <>
          <span className="loading"></span>
          Generating...
        </>
      ) : (
        <>
          <span className="btn-icon">⚡</span>
          Generate Text
        </>
      )}
    </button>
  )
}

export default GenerateButton

