function PromptInput({ prompt, setPrompt, theme, onKeyDown }) {
  return (
    <div className="form-group">
      <label htmlFor="prompt">Prompt:</label>
      <textarea
        id="prompt"
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={onKeyDown}
        rows={4}
        placeholder="Enter your prompt here..."
        className={theme}
      />
      <div className="hint">
        Press <kbd>Ctrl</kbd> + <kbd>Enter</kbd> to generate
      </div>
    </div>
  )
}

export default PromptInput

