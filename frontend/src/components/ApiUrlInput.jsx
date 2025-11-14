function ApiUrlInput({ apiUrl, setApiUrl, theme }) {
  return (
    <div className={`api-url ${theme}`}>
      <label>API URL:</label>
      <input
        type="text"
        value={apiUrl}
        onChange={(e) => setApiUrl(e.target.value)}
        placeholder="https://nima.eldertree.local"
      />
    </div>
  )
}

export default ApiUrlInput

