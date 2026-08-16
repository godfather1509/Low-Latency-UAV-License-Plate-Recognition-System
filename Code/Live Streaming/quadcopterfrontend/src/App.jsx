import { useState } from 'react'
import './App.css'
import WebCam from './components/webcam'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
    <WebCam/>
    </>
  )
}

export default App
