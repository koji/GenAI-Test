import { useState, useEffect } from 'react'
import { CreateMLCEngine } from '@mlc-ai/web-llm'
import './App.css'

// const LLM_MODEL = 'Llama-3.1-8B-Instruct-q4f32_1-MLC' // too slow
const LLM_MODEL = 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC'

function App() {
  const [engine, setEngine] = useState<any>(null) // engine
  const [inputText, setInputText] = useState<string>('') // input
  const [responseText, setResponseText] = useState<string>('') // output
  const [isLoading, setIsLoading] = useState<boolean>(false) // loading

  // init engine

  const initEngine = async () => {
    setIsLoading(true)
    try {
      const initProgressCallback = (initProgress: any) => {
        setResponseText(initProgress['text'])
      }
      const engine = await CreateMLCEngine(LLM_MODEL, {
        initProgressCallback: initProgressCallback,
      })
      setEngine(engine)
    } catch (error) {
      setResponseText('error')
      console.error(error)
    } finally {
      setIsLoading(false)
    }
    setIsLoading(false)
  }

  useEffect(() => {
    const init = async () => {
      await initEngine()
    }
    init()
  }, [])

  const handleClick = async () => {
    setIsLoading(true)
    try {
      const messages: any = [{ role: 'user', content: inputText }]
      const reply = await engine.chat.completions.create({
        messages,
      })
      if (reply.choices[0].message.content) {
        setResponseText(reply.choices[0].message.content)
      }
    } catch (error) {
      setResponseText('error')
      console.error(error)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div
      style={{
        display: 'flex',
        padding: '1rem',
        flexDirection: 'column',
        gap: '0.5rem',
      }}
    >
      <textarea
        value={inputText}
        onChange={(e) => setInputText(e.target.value)}
        rows={4}
        cols={50}
      />
      <button onClick={handleClick} disabled={isLoading}>
        {isLoading ? 'Loading...' : 'Send'}
      </button>
      <pre style={{ whiteSpace: 'pre-wrap' }}>{responseText}</pre>
    </div>
  )
}

export default App
