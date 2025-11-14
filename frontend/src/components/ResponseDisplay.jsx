function ResponseDisplay({ response, theme }) {
  return (
    <div className={`response ${theme}`}>
      <h3>Response</h3>
      <div className="response-text">{response}</div>
    </div>
  );
}

export default ResponseDisplay;
