import { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [feedbackStatus, setFeedbackStatus] = useState("");
  const [playingModel, setPlayingModel] = useState(null);
  const audioContextRef = useRef(null);
  const ttsAbortController = useRef(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const submitQuery = async (queryText) => {
    if (!queryText.trim() || isLoading) return;

    setMessages(prev => [...prev, { role: 'user', content: queryText }]);
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: queryText, session_id: "default_session" })
      });

      if (!response.ok) {
        const errText = await response.text();
        throw new Error(`Server returned ${response.status}: ${errText}`);
      }

      const data = await response.json();
      const aiMessage = { role: 'assistant', data: data, reqQuery: queryText };
      setMessages(prev => [...prev, aiMessage]);

    } catch (error) {
      console.error("Error fetching data:", error);
      setMessages(prev => [...prev, { role: 'assistant', error: `Backend Error: ${error.message}` }]);
    } finally {
      setIsLoading(false);
    }
  };

  const [recordingStream, setRecordingStream] = useState(null);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  const toggleRecording = async () => {
    if (isRecording) {
      // Stop the recording naturally
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stop();
      }
    } else {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        setRecordingStream(stream);
        
        const mediaRecorder = new MediaRecorder(stream);
        audioChunksRef.current = [];
        
        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };

        mediaRecorder.onstop = async () => {
          setIsRecording(false);
          const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
          const formData = new FormData();
          formData.append("file", audioBlob, "audio.webm");

          // Shut off the microphone completely
          stream.getTracks().forEach(track => track.stop());
          setRecordingStream(null);

          try {
            setIsTranscribing(true);
            const response = await fetch("http://localhost:8000/transcribe", {
              method: "POST",
              body: formData
            });
            if (!response.ok) {
              const errData = await response.json();
              throw new Error(errData.detail || "Transcription Failed");
            }
            
            const data = await response.json();
            if (data.text && data.text.trim()) {
              setInput(data.text);
            }
          } catch (e) {
            console.error("Whisper Error:", e);
            alert("Transcription failed: " + e.message);
          } finally {
            setIsTranscribing(false);
          }
        };

        mediaRecorder.start();
        mediaRecorderRef.current = mediaRecorder;
        setIsRecording(true);
        setInput(""); // Ensure text box is clean for wave effect
      } catch (err) {
        console.error("Microphone Error", err);
        alert("Microphone access denied or completely unavailable.");
      }
    }
  };

  const submitFeedback = async (question, selectedModel) => {
    try {
      setFeedbackStatus("Submitting...");
      const response = await fetch('http://localhost:8000/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: question, selected_model: selectedModel })
      });
      if (response.ok) {
        setFeedbackStatus(`Saved! You voted for ${selectedModel}.`);
      } else {
        setFeedbackStatus("Failed to submit feedback.");
      }
    } catch (error) {
      setFeedbackStatus(`Error submitting feedback: ${error.message}`);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;
    const currentQuery = input;
    setInput('');
    await submitQuery(currentQuery);
  };

  return (
    <div className="app-container">
      <div className="glass-panel main-chat">
        <header className="app-header">
          <h1>🔬 AI Research Assistant</h1>
          <p>Powered by Llama 3.2 for local RAG, Claude, OpenAI, and Google Gemini</p>
        </header>

        <div className="messages-area">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message-wrapper ${msg.role}`}>
              {msg.role === 'user' ? (
                <div className="glass-bubble user-bubble">
                  {msg.content}
                </div>
              ) : (
                <div className="assistant-bubble-container">
                  {msg.error ? (
                    <div className="glass-bubble error-bubble">
                      {msg.error}
                    </div>
                  ) : (
                    <>

                      <div className="models-comparison">
                        <div className="glass-bubble model-bubble llama-bubble">
                          <h4>
                            🦙 AI Research Assistant (Llama 3.2)
                          </h4>
                          <div className="bubble-scroll-area">
                            <div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.data.llama_response || "No response generated."}</ReactMarkdown></div>
                            {msg.data.sources && msg.data.sources.length > 0 && !msg.data.llama_response.match(/(cannot answer this|don't see any context)/i) && (
                              <div className="llama-sources-box" style={{marginTop: '15px', padding: '10px', background: 'rgba(0,0,0,0.2)', borderRadius: '8px', fontSize: '0.85rem', textAlign: 'left'}}>
                                <strong style={{color: '#a7f3d0'}}>Sources used by Llama 3.2:</strong>
                                <ul style={{margin: '5px 0 0 20px', padding: 0}}>
                                  {msg.data.sources.map((s, i) => <li style={{margin: '3px 0'}} key={i}>{s}</li>)}
                                </ul>
                              </div>
                            )}
                          </div>
                        </div>
                        <div className="glass-bubble model-bubble claude-bubble">
                          <h4>
                            🧠 Anthropic (claude-sonnet-4-6)
                          </h4>
                          <div className="bubble-scroll-area">
                            <div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.data.claude_response || "No response generated."}</ReactMarkdown></div>
                          </div>
                        </div>
                        <div className="glass-bubble model-bubble gpt-bubble">
                          <h4>
                            🤖 OpenAI (gpt-4o)
                          </h4>
                          <div className="bubble-scroll-area">
                            <div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.data.gpt_response || "No response generated."}</ReactMarkdown></div>
                          </div>
                        </div>
                        <div className="glass-bubble model-bubble gemini-bubble">
                          <h4>
                            ✨ Google (gemini-flash-latest)
                          </h4>
                          <div className="bubble-scroll-area">
                            <div className="markdown-body"><ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.data.gemini_response || "No response generated."}</ReactMarkdown></div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="feedback-container">
                        <p>For this particular question, which response do you find very accurate?</p>
                        <div className="feedback-buttons">
                          <button className="btn-feedback" onClick={() => submitFeedback(msg.reqQuery, "Llama 3.2")}>Llama 3.2</button>
                          <button className="btn-feedback" onClick={() => submitFeedback(msg.reqQuery, "Claude")}>Claude</button>
                          <button className="btn-feedback" onClick={() => submitFeedback(msg.reqQuery, "GPT-4o")}>GPT-4o</button>
                          <button className="btn-feedback" onClick={() => submitFeedback(msg.reqQuery, "Gemini")}>Gemini</button>
                        </div>
                        {feedbackStatus && <p style={{ fontSize: "0.9rem", color: "#a7f3d0", marginTop: "10px" }}>{feedbackStatus}</p>}
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          ))}
          
          {isLoading && (
            <div className="message-wrapper assistant">
              <div className="loading-indicator">
                <div className="dot"></div><div className="dot"></div><div className="dot"></div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form className="input-form" onSubmit={handleSubmit}>
          <div className="input-wrapper">
            <input
              type="text"
              className="glass-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question or speak..."
              disabled={isLoading || isRecording || isTranscribing}
            />
            {isRecording && (
              <div className="wave-overlay">
                <div className="wave-bars">
                  <div className="bar"></div>
                  <div className="bar"></div>
                  <div className="bar"></div>
                  <div className="bar"></div>
                  <div className="bar"></div>
                </div>
                <span className="interim-text">{input || "Listening..."}</span>
              </div>
            )}
            {isTranscribing && (
              <div className="wave-overlay" style={{justifyContent: 'flex-start'}}>
                <span className="interim-text">Converting...</span>
                <div className="loading-indicator" style={{background: 'transparent', padding: '0 10px'}}>
                  <div className="dot" style={{width: '6px', height: '6px'}}></div>
                  <div className="dot" style={{width: '6px', height: '6px'}}></div>
                  <div className="dot" style={{width: '6px', height: '6px'}}></div>
                </div>
              </div>
            )}
            <button 
              type="button"
              className={`mic-btn-inside ${isRecording ? 'recording' : ''}`}
              onClick={toggleRecording}
              title={isRecording ? "Stop Dictation" : "Start Dictation"}
            >
              {isRecording ? '🛑' : '🎤'}
            </button>
          </div>
          <button type="submit" className="glass-button" disabled={isLoading || isRecording || isTranscribing || !input.trim()}>
            {isLoading ? '...' : 'Send'}
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
