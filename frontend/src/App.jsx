import { useState, useEffect } from 'react'
import Header from './components/Header'
import PromptInput from './components/PromptInput'
import ParametersPanel from './components/ParametersPanel'
import GenerateButton from './components/GenerateButton'
import ResponseDisplay from './components/ResponseDisplay'
import StatusMessage from './components/StatusMessage'
import ApiUrlInput from './components/ApiUrlInput'
import { useTheme } from './hooks/useTheme'
import './App.css'

function App() {
  const { theme, toggleTheme } = useTheme()
  const [apiUrl, setApiUrl] = useState('https://nima.eldertree.local')
  const [prompt, setPrompt] = useState('What is Kubernetes?')
  const [maxLength, setMaxLength] = useState(200)
  const [temperature, setTemperature] = useState(0.8)
  const [topK, setTopK] = useState(50)
  const [response, setResponse] = useState(null)
  const [status, setStatus] = useState(null)
  const [isLoading, setIsLoading] = useState(false)

  const generateText = async () => {
    if (!prompt.trim()) {
      setStatus({ type: 'error', message: 'Please enter a prompt' })
      return
    }

    setIsLoading(true)
    setStatus(null)
    setResponse(null)

    try {
      // Health check
      setStatus({ type: 'info', message: 'Checking API health...' })
      const healthUrl = `${apiUrl}/health`
      const healthResponse = await fetch(healthUrl)
      
      if (!healthResponse.ok) {
        throw new Error(`Health check failed: ${healthResponse.status}`)
      }

      const healthData = await healthResponse.json()
      if (!healthData.model_loaded || !healthData.tokenizer_loaded) {
        throw new Error('Model or tokenizer not loaded')
      }

      setStatus({ type: 'info', message: 'Generating text...' })

      // Inference request
      const inferenceUrl = `${apiUrl}/v1/inference`
      const response = await fetch(inferenceUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          prompt: prompt,
          max_length: maxLength,
          temperature: temperature,
          top_k: topK
        })
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: response.statusText }))
        throw new Error(errorData.detail || `Request failed: ${response.status}`)
      }

      const data = await response.json()
      setResponse(data.response || 'No response generated')
      setStatus({ type: 'success', message: 'Success! Text generated.' })

    } catch (error) {
      console.error('Error:', error)
      setStatus({ type: 'error', message: `Error: ${error.message}` })
      setResponse(null)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className={`app ${theme}`}>
      <div className="container">
        <Header theme={theme} toggleTheme={toggleTheme} />
        
        <ApiUrlInput 
          apiUrl={apiUrl} 
          setApiUrl={setApiUrl}
          theme={theme}
        />

        <PromptInput 
          prompt={prompt}
          setPrompt={setPrompt}
          theme={theme}
          onKeyDown={(e) => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
              generateText()
            }
          }}
        />

        <ParametersPanel
          maxLength={maxLength}
          setMaxLength={setMaxLength}
          temperature={temperature}
          setTemperature={setTemperature}
          topK={topK}
          setTopK={setTopK}
          theme={theme}
        />

        <GenerateButton
          onClick={generateText}
          isLoading={isLoading}
          theme={theme}
        />

        <StatusMessage status={status} theme={theme} />

        {response && (
          <ResponseDisplay response={response} theme={theme} />
        )}
      </div>
    </div>
  )
}

export default App

