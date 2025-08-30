import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Loader2 } from "lucide-react";

function App() {
  const [messages, setMessages] = useState([
    { role: 'bot', text: '🧠 Xin chào! Tôi là chatbot của bạn.' }
  ]);
  const [input, setInput] = useState('');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [botTypingText, setBotTypingText] = useState('');

  const chatBoxRef = useRef(null); // ✅ ref cho chat box

  // ✅ Mỗi khi messages hoặc botTypingText thay đổi → scroll xuống cuối
  useEffect(() => {
    if (chatBoxRef.current) {
      chatBoxRef.current.scrollTop = chatBoxRef.current.scrollHeight;
    }
  }, [messages, botTypingText]);

  const handleSend = async () => {
    if (input.trim() === '' && !file) return;

    const newMessages = [...messages];
    if (input.trim() !== '') {
      newMessages.push({ role: 'user', text: input });
    }

    setMessages(newMessages);
    setLoading(true);

    const formData = new FormData();
    formData.append("message", input);
    if (file) {
      formData.append("file", file);
    }

    setInput('');
    setFile(null);

    try {
      const res = await axios.post("http://127.0.0.1:8000/chat", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      const data = res.data;
      simulateTyping(data.response, () => {
        setMessages(prev => [...prev, { role: 'bot', text: data.response }]);
        setBotTypingText('');
      });
    } catch (error) {
      setMessages(prev => [...prev, { role: 'bot', text: '❌ Lỗi kết nối server' }]);
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (!selected) return;

    const allowedExtensions = ['doc', 'docx','pdf','csv','jpg','png'];
    const extension = selected.name.split('.').pop().toLowerCase();
    if (!allowedExtensions.includes(extension)) {
      alert('❌ Chỉ chấp nhận file .doc, .docx, .pdf hoặc .csv, .png, .jpg');
      return;
    }

    setFile(selected);
  };

  const simulateTyping = (fullText, callback) => {
    let index = 0;
    setBotTypingText('');
    const interval = setInterval(() => {
      if (index < fullText.length) {
        setBotTypingText(prev => prev + fullText[index]);
        index++;
      } else {
        clearInterval(interval);
        callback();
      }
    }, 1); // 30ms mỗi ký tự
  };

  return (
    <div className="app">
      <h1>🧠 LLM Chatbot</h1>
      <div className="chat-box" ref={chatBoxRef}>
        {messages.map((msg, idx) => (
          <div key={idx} className={`msg ${msg.role}`}>
            <strong>{msg.role === 'user' ? 'Bạn' : 'Bot'}:</strong>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.text}</ReactMarkdown>
          </div>
        ))}
        {botTypingText && (
          <div className="msg bot">
            <strong>Bot:</strong> <ReactMarkdown>{botTypingText}</ReactMarkdown>
          </div>
        )}
        {loading && (
      <div className="msg bot">
        <div className="flex items-center gap-2 bg-green-50 p-3 rounded-xl shadow-sm">
          <strong className="text-green-700">Bot:</strong>
          <Loader2 className="loading-icon" />
          <span>Đang xử lý, vui lòng chờ...</span>
        </div>
      </div>
    )}
      </div>
      <div className="input-area">
        <input
          type="text"
          placeholder="Nhập tin nhắn..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
        />
        <input
          type="file"
          accept=".doc,.docx,.csv,.pdf,.png,.jpg"
          onChange={handleFileChange}
          style={{ marginLeft: '10px' }}
        />
        <button onClick={handleSend}>Gửi</button>
      </div>
    </div>
  );
}

export default App;
