function ParametersPanel({ maxLength, setMaxLength, temperature, setTemperature, topK, setTopK, theme }) {
  return (
    <div className={`params ${theme}`}>
      <div className="param-group">
        <label htmlFor="maxLength">Max Length</label>
        <input
          type="range"
          id="maxLength"
          min="50"
          max="500"
          value={maxLength}
          onChange={(e) => setMaxLength(parseInt(e.target.value))}
          step="10"
        />
        <span className="param-value">{maxLength}</span>
      </div>

      <div className="param-group">
        <label htmlFor="temperature">Temperature</label>
        <input
          type="range"
          id="temperature"
          min="0.1"
          max="2.0"
          value={temperature}
          onChange={(e) => setTemperature(parseFloat(e.target.value))}
          step="0.1"
        />
        <span className="param-value">{temperature.toFixed(1)}</span>
      </div>

      <div className="param-group">
        <label htmlFor="topK">Top K</label>
        <input
          type="range"
          id="topK"
          min="1"
          max="100"
          value={topK}
          onChange={(e) => setTopK(parseInt(e.target.value))}
          step="1"
        />
        <span className="param-value">{topK}</span>
      </div>
    </div>
  )
}

export default ParametersPanel

