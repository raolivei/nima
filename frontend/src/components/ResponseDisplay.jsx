function ResponseDisplay({ response, theme }) {
  return (
    <div className={`response ${theme}`}>
      <h3>Generated Response:</h3>
      <div className="response-text">{response}</div>
    </div>
  )
}

export default ResponseDisplay

