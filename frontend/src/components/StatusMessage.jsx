function StatusMessage({ status, theme }) {
  if (!status) return null

  return (
    <div className={`status ${status.type} ${theme}`}>
      {status.type === 'success' && <span className="status-icon">✓</span>}
      {status.type === 'error' && <span className="status-icon">✗</span>}
      {status.type === 'info' && <span className="status-icon">ℹ</span>}
      <span>{status.message}</span>
    </div>
  )
}

export default StatusMessage

