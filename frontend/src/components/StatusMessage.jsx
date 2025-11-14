function StatusMessage({ status, theme }) {
  if (!status) return null;

  return (
    <div className={`status ${status.type} ${theme}`}>
      <span>{status.message}</span>
    </div>
  );
}

export default StatusMessage;
