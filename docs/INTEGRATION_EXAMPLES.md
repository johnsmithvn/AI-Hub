# 🤝 **Integration Examples**

## 📋 **Overview**

Comprehensive integration examples cho AI Backend Hub với các popular frameworks và applications. Tất cả examples đều sử dụng OpenAI-compatible API format cho seamless migration.

---

## ⚛️ **React/Next.js Integration**

### **1. React ChatBot Component**

```typescript
// hooks/useAIHub.ts
import { useState, useCallback } from 'react';

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp?: Date;
}

interface UseAIHubConfig {
  baseUrl?: string;
  apiKey?: string;
  defaultModel?: string;
}

export const useAIHub = (config: UseAIHubConfig = {}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const baseUrl = config.baseUrl || process.env.NEXT_PUBLIC_AI_HUB_URL || 'http://localhost:8000';
  const apiKey = config.apiKey || process.env.NEXT_PUBLIC_AI_HUB_API_KEY || 'local-api-key';
  const defaultModel = config.defaultModel || 'llama2-7b-chat';

  const sendMessage = useCallback(async (
    content: string, 
    model?: string,
    systemPrompt?: string
  ) => {
    setIsLoading(true);
    setError(null);

    const userMessage: ChatMessage = {
      role: 'user',
      content,
      timestamp: new Date()
    };

    const updatedMessages = [...messages, userMessage];
    setMessages(updatedMessages);

    try {
      const requestMessages = systemPrompt 
        ? [{ role: 'system', content: systemPrompt }, ...updatedMessages]
        : updatedMessages;

      const response = await fetch(`${baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${apiKey}`,
        },
        body: JSON.stringify({
          model: model || defaultModel,
          messages: requestMessages.map(msg => ({
            role: msg.role,
            content: msg.content
          })),
          max_tokens: 2048,
          temperature: 0.7,
          stream: false
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: data.choices[0].message.content,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
      return assistantMessage;

    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      throw err;
    } finally {
      setIsLoading(false);
    }
  }, [messages, baseUrl, apiKey, defaultModel]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    sendMessage,
    clearMessages
  };
};
```

```typescript
// components/ChatInterface.tsx
import React, { useState } from 'react';
import { useAIHub } from '../hooks/useAIHub';

const ChatInterface: React.FC = () => {
  const [input, setInput] = useState('');
  const [selectedModel, setSelectedModel] = useState('llama2-7b-chat');
  const { messages, isLoading, error, sendMessage, clearMessages } = useAIHub();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    try {
      await sendMessage(input, selectedModel);
      setInput('');
    } catch (error) {
      console.error('Failed to send message:', error);
    }
  };

  const modelOptions = [
    { value: 'llama2-7b-chat', label: 'Llama 2 Chat' },
    { value: 'codellama-7b-instruct', label: 'Code Llama' },
    { value: 'qwen-7b-vietnamese', label: 'Qwen Vietnamese' },
  ];

  return (
    <div className="max-w-4xl mx-auto p-4">
      <div className="bg-white rounded-lg shadow-lg">
        {/* Header */}
        <div className="border-b p-4">
          <div className="flex justify-between items-center">
            <h1 className="text-xl font-bold">AI Backend Hub Chat</h1>
            <div className="flex gap-2">
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="border rounded px-3 py-1"
              >
                {modelOptions.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
              <button
                onClick={clearMessages}
                className="px-3 py-1 bg-gray-500 text-white rounded hover:bg-gray-600"
              >
                Clear
              </button>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="h-96 overflow-y-auto p-4 space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                  message.role === 'user'
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 text-gray-800'
                }`}
              >
                <div className="text-sm">{message.content}</div>
                {message.timestamp && (
                  <div className="text-xs opacity-75 mt-1">
                    {message.timestamp.toLocaleTimeString()}
                  </div>
                )}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gray-200 text-gray-800 px-4 py-2 rounded-lg">
                <div className="flex items-center space-x-2">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600"></div>
                  <span>AI is thinking...</span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Input */}
        <form onSubmit={handleSubmit} className="border-t p-4">
          <div className="flex space-x-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Type your message..."
              className="flex-1 border rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50"
            >
              Send
            </button>
          </div>
          {error && (
            <div className="text-red-500 text-sm mt-2">
              Error: {error}
            </div>
          )}
        </form>
      </div>
    </div>
  );
};

export default ChatInterface;
```

### **2. Next.js API Routes**

```typescript
// pages/api/chat.ts (Pages Router)
// hoặc app/api/chat/route.ts (App Router)

import type { NextApiRequest, NextApiResponse } from 'next';

interface ChatRequest {
  message: string;
  model?: string;
  systemPrompt?: string;
}

interface ChatResponse {
  reply: string;
  model: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<ChatResponse | { error: string }>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { message, model = 'llama2-7b-chat', systemPrompt }: ChatRequest = req.body;

  if (!message) {
    return res.status(400).json({ error: 'Message is required' });
  }

  try {
    const messages = systemPrompt
      ? [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: message }
        ]
      : [{ role: 'user', content: message }];

    const response = await fetch(`${process.env.AI_HUB_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.AI_HUB_API_KEY}`,
      },
      body: JSON.stringify({
        model,
        messages,
        max_tokens: 2048,
        temperature: 0.7,
      }),
    });

    if (!response.ok) {
      throw new Error(`AI Hub API error: ${response.status}`);
    }

    const data = await response.json();

    res.status(200).json({
      reply: data.choices[0].message.content,
      model: data.model,
      usage: data.usage,
    });

  } catch (error) {
    console.error('Chat API error:', error);
    res.status(500).json({ 
      error: error instanceof Error ? error.message : 'Internal server error' 
    });
  }
}
```

---

## 🐍 **Python Application Integration**

### **1. FastAPI Integration**

```python
# ai_client.py
import asyncio
import httpx
from typing import List, Dict, Optional
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str

class AIHubClient:
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = "local-api-key"):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session: Optional[httpx.AsyncClient] = None
    
    async def __aenter__(self):
        self.session = httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=300.0
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.aclose()
    
    async def chat_completion(
        self,
        messages: List[ChatMessage],
        model: str = "llama2-7b-chat",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Dict:
        """Send chat completion request"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        
        payload = {
            "model": model,
            "messages": [msg.dict() for msg in messages],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        
        response = await self.session.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()
    
    async def list_models(self) -> List[Dict]:
        """Get available models"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        
        response = await self.session.get("/v1/models")
        response.raise_for_status()
        return response.json()["data"]
    
    async def load_model(self, model_name: str, quantization: str = "4bit") -> bool:
        """Load a specific model"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        
        payload = {
            "model_name": model_name,
            "quantization": quantization
        }
        
        response = await self.session.post("/v1/models/load", json=payload)
        if response.status_code == 200:
            return response.json().get("success", False)
        return False
    
    async def start_training(
        self,
        base_model: str,
        dataset_path: str,
        output_dir: str,
        config: Dict
    ) -> str:
        """Start a training job"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use 'async with' context.")
        
        payload = {
            "base_model": base_model,
            "dataset_path": dataset_path,
            "output_dir": output_dir,
            "config": config
        }
        
        response = await self.session.post("/v1/training/jobs", json=payload)
        response.raise_for_status()
        return response.json()["job_id"]
```

```python
# main.py - FastAPI Application
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import asyncio

from ai_client import AIHubClient, ChatMessage

app = FastAPI(title="My App with AI Backend Hub")

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "llama2-7b-chat"
    context: Optional[List[str]] = None

class ChatResponse(BaseModel):
    reply: str
    model: str
    context_used: bool

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint using AI Backend Hub"""
    
    try:
        async with AIHubClient() as ai_client:
            # Prepare messages
            messages = []
            
            # Add context if provided
            if request.context:
                context_content = "\n".join(request.context)
                messages.append(ChatMessage(
                    role="system",
                    content=f"Use this context to answer questions:\n{context_content}"
                ))
            
            messages.append(ChatMessage(
                role="user",
                content=request.message
            ))
            
            # Get AI response
            response = await ai_client.chat_completion(
                messages=messages,
                model=request.model
            )
            
            reply = response["choices"][0]["message"]["content"]
            
            return ChatResponse(
                reply=reply,
                model=response["model"],
                context_used=bool(request.context)
            )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """List available AI models"""
    try:
        async with AIHubClient() as ai_client:
            models = await ai_client.list_models()
            return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/training")
async def start_training_job(
    base_model: str,
    dataset_path: str,
    background_tasks: BackgroundTasks
):
    """Start training job"""
    try:
        async with AIHubClient() as ai_client:
            job_id = await ai_client.start_training(
                base_model=base_model,
                dataset_path=dataset_path,
                output_dir=f"./trained_models/{base_model}_custom",
                config={
                    "epochs": 3,
                    "batch_size": 4,
                    "learning_rate": 2e-4
                }
            )
            
            return {"job_id": job_id, "status": "started"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)
```

### **2. Django Integration**

```python
# ai_hub/client.py
import requests
from django.conf import settings
from typing import List, Dict

class AIHubClient:
    def __init__(self):
        self.base_url = getattr(settings, 'AI_HUB_URL', 'http://localhost:8000')
        self.api_key = getattr(settings, 'AI_HUB_API_KEY', 'local-api-key')
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
    
    def chat_completion(self, messages: List[Dict], model: str = "llama2-7b-chat") -> str:
        """Send chat completion request"""
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.7
        }
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            headers=self.headers,
            timeout=60
        )
        response.raise_for_status()
        
        data = response.json()
        return data["choices"][0]["message"]["content"]

# ai_hub/views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json

from .client import AIHubClient

@csrf_exempt
@require_http_methods(["POST"])
def chat_view(request):
    """Django chat view"""
    try:
        data = json.loads(request.body)
        message = data.get('message')
        model = data.get('model', 'llama2-7b-chat')
        
        if not message:
            return JsonResponse({'error': 'Message is required'}, status=400)
        
        client = AIHubClient()
        response = client.chat_completion(
            messages=[{"role": "user", "content": message}],
            model=model
        )
        
        return JsonResponse({
            'reply': response,
            'model': model
        })
    
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

# models.py
from django.db import models

class Conversation(models.Model):
    user_message = models.TextField()
    ai_response = models.TextField()
    model_used = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']

# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('api/chat/', views.chat_view, name='chat'),
]
```

---

## 🟢 **Node.js/Express Integration**

### **1. Express API Server**

```javascript
// ai-client.js
const axios = require('axios');

class AIHubClient {
  constructor(baseUrl = 'http://localhost:8000', apiKey = 'local-api-key') {
    this.baseUrl = baseUrl;
    this.apiKey = apiKey;
    this.client = axios.create({
      baseURL: baseUrl,
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      timeout: 60000
    });
  }

  async chatCompletion(messages, model = 'llama2-7b-chat', options = {}) {
    try {
      const response = await this.client.post('/v1/chat/completions', {
        model,
        messages,
        max_tokens: options.maxTokens || 2048,
        temperature: options.temperature || 0.7,
        stream: options.stream || false
      });
      
      return response.data;
    } catch (error) {
      throw new Error(`AI Hub API error: ${error.message}`);
    }
  }

  async listModels() {
    try {
      const response = await this.client.get('/v1/models');
      return response.data.data;
    } catch (error) {
      throw new Error(`Failed to list models: ${error.message}`);
    }
  }

  async loadModel(modelName, quantization = '4bit') {
    try {
      const response = await this.client.post('/v1/models/load', {
        model_name: modelName,
        quantization
      });
      return response.data.success;
    } catch (error) {
      throw new Error(`Failed to load model: ${error.message}`);
    }
  }
}

module.exports = AIHubClient;
```

```javascript
// app.js
const express = require('express');
const cors = require('cors');
const AIHubClient = require('./ai-client');

const app = express();
const aiClient = new AIHubClient();

app.use(cors());
app.use(express.json({ limit: '10mb' }));

// Chat endpoint
app.post('/api/chat', async (req, res) => {
  try {
    const { message, model = 'llama2-7b-chat', context } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }

    // Prepare messages
    const messages = [];
    
    if (context) {
      messages.push({
        role: 'system',
        content: `Context: ${context}`
      });
    }
    
    messages.push({
      role: 'user',
      content: message
    });

    const response = await aiClient.chatCompletion(messages, model);
    
    res.json({
      reply: response.choices[0].message.content,
      model: response.model,
      usage: response.usage
    });

  } catch (error) {
    console.error('Chat error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Models endpoint
app.get('/api/models', async (req, res) => {
  try {
    const models = await aiClient.listModels();
    res.json({ models });
  } catch (error) {
    console.error('Models error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Code generation endpoint
app.post('/api/code', async (req, res) => {
  try {
    const { prompt, language = 'python' } = req.body;
    
    const systemPrompt = `You are a code generation assistant. Generate clean, well-commented ${language} code.`;
    
    const response = await aiClient.chatCompletion([
      { role: 'system', content: systemPrompt },
      { role: 'user', content: prompt }
    ], 'codellama-7b-instruct');
    
    res.json({
      code: response.choices[0].message.content,
      language
    });

  } catch (error) {
    console.error('Code generation error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Vietnamese chat endpoint
app.post('/api/chat/vietnamese', async (req, res) => {
  try {
    const { message } = req.body;
    
    const systemPrompt = "Bạn là một trợ lý AI thông minh, trả lời bằng tiếng Việt một cách tự nhiên và hữu ích.";
    
    const response = await aiClient.chatCompletion([
      { role: 'system', content: systemPrompt },
      { role: 'user', content: message }
    ], 'qwen-7b-vietnamese');
    
    res.json({
      reply: response.choices[0].message.content,
      language: 'vietnamese'
    });

  } catch (error) {
    console.error('Vietnamese chat error:', error);
    res.status(500).json({ error: error.message });
  }
});

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date().toISOString() });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
```

---

## 📱 **Mobile Integration**

### **1. React Native**

```typescript
// AIHubService.ts
export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}

export interface ChatResponse {
  reply: string;
  model: string;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

class AIHubService {
  private baseUrl: string;
  private apiKey: string;

  constructor(baseUrl = 'http://localhost:8000', apiKey = 'local-api-key') {
    this.baseUrl = baseUrl.replace(/\/$/, '');
    this.apiKey = apiKey;
  }

  async chatCompletion(
    messages: ChatMessage[],
    model = 'llama2-7b-chat',
    options: {
      maxTokens?: number;
      temperature?: number;
    } = {}
  ): Promise<ChatResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/chat/completions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.apiKey}`,
        },
        body: JSON.stringify({
          model,
          messages,
          max_tokens: options.maxTokens || 2048,
          temperature: options.temperature || 0.7,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      return {
        reply: data.choices[0].message.content,
        model: data.model,
        usage: data.usage,
      };
    } catch (error) {
      throw new Error(`AI Hub API error: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  async listModels(): Promise<any[]> {
    try {
      const response = await fetch(`${this.baseUrl}/v1/models`, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
        },
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data.data || [];
    } catch (error) {
      throw new Error(`Failed to list models: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }
}

export default AIHubService;
```

```typescript
// ChatScreen.tsx
import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  FlatList,
  StyleSheet,
  Alert,
  ActivityIndicator
} from 'react-native';
import AIHubService, { ChatMessage } from '../services/AIHubService';

const ChatScreen: React.FC = () => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [aiService] = useState(() => new AIHubService());

  const sendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    const userMessage: ChatMessage = {
      role: 'user',
      content: inputText.trim(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setIsLoading(true);

    try {
      const response = await aiService.chatCompletion([...messages, userMessage]);
      
      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.reply,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      Alert.alert('Error', error instanceof Error ? error.message : 'Unknown error');
    } finally {
      setIsLoading(false);
    }
  };

  const renderMessage = ({ item }: { item: ChatMessage }) => (
    <View style={[
      styles.messageContainer,
      item.role === 'user' ? styles.userMessage : styles.assistantMessage
    ]}>
      <Text style={[
        styles.messageText,
        item.role === 'user' ? styles.userText : styles.assistantText
      ]}>
        {item.content}
      </Text>
    </View>
  );

  return (
    <View style={styles.container}>
      <FlatList
        data={messages}
        renderItem={renderMessage}
        keyExtractor={(_, index) => index.toString()}
        style={styles.messagesList}
      />
      
      {isLoading && (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="small" color="#007AFF" />
          <Text style={styles.loadingText}>AI is thinking...</Text>
        </View>
      )}
      
      <View style={styles.inputContainer}>
        <TextInput
          style={styles.textInput}
          value={inputText}
          onChangeText={setInputText}
          placeholder="Type your message..."
          multiline
          editable={!isLoading}
        />
        <TouchableOpacity
          style={[styles.sendButton, (!inputText.trim() || isLoading) && styles.disabledButton]}
          onPress={sendMessage}
          disabled={!inputText.trim() || isLoading}
        >
          <Text style={styles.sendButtonText}>Send</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f5f5f5',
  },
  messagesList: {
    flex: 1,
    padding: 16,
  },
  messageContainer: {
    marginVertical: 4,
    padding: 12,
    borderRadius: 12,
    maxWidth: '80%',
  },
  userMessage: {
    backgroundColor: '#007AFF',
    alignSelf: 'flex-end',
  },
  assistantMessage: {
    backgroundColor: '#ffffff',
    alignSelf: 'flex-start',
    borderWidth: 1,
    borderColor: '#e0e0e0',
  },
  messageText: {
    fontSize: 16,
  },
  userText: {
    color: '#ffffff',
  },
  assistantText: {
    color: '#333333',
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
  },
  loadingText: {
    marginLeft: 8,
    color: '#666666',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 16,
    backgroundColor: '#ffffff',
    borderTopWidth: 1,
    borderTopColor: '#e0e0e0',
  },
  textInput: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#e0e0e0',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 8,
    maxHeight: 100,
    marginRight: 8,
  },
  sendButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 20,
    justifyContent: 'center',
  },
  disabledButton: {
    backgroundColor: '#cccccc',
  },
  sendButtonText: {
    color: '#ffffff',
    fontWeight: 'bold',
  },
});

export default ChatScreen;
```

### **2. Flutter Integration**

```dart
// ai_hub_service.dart
import 'dart:convert';
import 'package:http/http.dart' as http;

class ChatMessage {
  final String role;
  final String content;

  ChatMessage({required this.role, required this.content});

  Map<String, dynamic> toJson() => {
    'role': role,
    'content': content,
  };
}

class AIHubService {
  final String baseUrl;
  final String apiKey;
  final http.Client _client = http.Client();

  AIHubService({
    this.baseUrl = 'http://localhost:8000',
    this.apiKey = 'local-api-key',
  });

  Future<String> chatCompletion({
    required List<ChatMessage> messages,
    String model = 'llama2-7b-chat',
    int maxTokens = 2048,
    double temperature = 0.7,
  }) async {
    try {
      final response = await _client.post(
        Uri.parse('$baseUrl/v1/chat/completions'),
        headers: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer $apiKey',
        },
        body: jsonEncode({
          'model': model,
          'messages': messages.map((m) => m.toJson()).toList(),
          'max_tokens': maxTokens,
          'temperature': temperature,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['choices'][0]['message']['content'];
      } else {
        throw Exception('AI Hub API error: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Failed to get AI response: $e');
    }
  }

  Future<List<dynamic>> listModels() async {
    try {
      final response = await _client.get(
        Uri.parse('$baseUrl/v1/models'),
        headers: {
          'Authorization': 'Bearer $apiKey',
        },
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        return data['data'] ?? [];
      } else {
        throw Exception('Failed to list models: ${response.statusCode}');
      }
    } catch (e) {
      throw Exception('Failed to list models: $e');
    }
  }

  void dispose() {
    _client.close();
  }
}
```

```dart
// chat_screen.dart
import 'package:flutter/material.dart';
import 'ai_hub_service.dart';

class ChatScreen extends StatefulWidget {
  @override
  _ChatScreenState createState() => _ChatScreenState();
}

class _ChatScreenState extends State<ChatScreen> {
  final AIHubService _aiService = AIHubService();
  final TextEditingController _controller = TextEditingController();
  final List<ChatMessage> _messages = [];
  bool _isLoading = false;

  Future<void> _sendMessage() async {
    if (_controller.text.trim().isEmpty || _isLoading) return;

    final userMessage = ChatMessage(
      role: 'user',
      content: _controller.text.trim(),
    );

    setState(() {
      _messages.add(userMessage);
      _isLoading = true;
    });

    final inputText = _controller.text;
    _controller.clear();

    try {
      final response = await _aiService.chatCompletion(
        messages: _messages,
        model: 'llama2-7b-chat',
      );

      setState(() {
        _messages.add(ChatMessage(
          role: 'assistant',
          content: response,
        ));
      });
    } catch (e) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: $e')),
      );
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('AI Backend Hub Chat'),
        backgroundColor: Colors.blue,
      ),
      body: Column(
        children: [
          Expanded(
            child: ListView.builder(
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final message = _messages[index];
                final isUser = message.role == 'user';
                
                return Container(
                  margin: EdgeInsets.symmetric(vertical: 4, horizontal: 8),
                  child: Row(
                    mainAxisAlignment: isUser 
                        ? MainAxisAlignment.end 
                        : MainAxisAlignment.start,
                    children: [
                      Container(
                        constraints: BoxConstraints(
                          maxWidth: MediaQuery.of(context).size.width * 0.8,
                        ),
                        padding: EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: isUser ? Colors.blue : Colors.grey[200],
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          message.content,
                          style: TextStyle(
                            color: isUser ? Colors.white : Colors.black,
                          ),
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
          ),
          if (_isLoading)
            Padding(
              padding: EdgeInsets.all(16),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(width: 12),
                  Text('AI is thinking...'),
                ],
              ),
            ),
          Container(
            padding: EdgeInsets.all(8),
            decoration: BoxDecoration(
              border: Border(top: BorderSide(color: Colors.grey[300]!)),
            ),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    decoration: InputDecoration(
                      hintText: 'Type your message...',
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(20),
                      ),
                    ),
                    maxLines: null,
                    enabled: !_isLoading,
                  ),
                ),
                SizedBox(width: 8),
                IconButton(
                  onPressed: _isLoading ? null : _sendMessage,
                  icon: Icon(Icons.send),
                  color: Colors.blue,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _aiService.dispose();
    _controller.dispose();
    super.dispose();
  }
}
```

---

## 🔗 **Advanced Integration Patterns**

### **1. WebSocket Real-time Integration**

```javascript
// websocket-client.js
class AIHubWebSocketClient {
  constructor(baseUrl = 'ws://localhost:8000') {
    this.baseUrl = baseUrl;
    this.ws = null;
    this.messageHandlers = new Map();
  }

  connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(`${this.baseUrl}/ws`);
      
      this.ws.onopen = () => {
        console.log('Connected to AI Hub WebSocket');
        resolve();
      };
      
      this.ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      };
      
      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      };
      
      this.ws.onclose = () => {
        console.log('Disconnected from AI Hub WebSocket');
      };
    });
  }

  handleMessage(data) {
    const handler = this.messageHandlers.get(data.type);
    if (handler) {
      handler(data);
    }
  }

  onMessage(type, handler) {
    this.messageHandlers.set(type, handler);
  }

  sendStreamingChat(messages, model = 'llama2-7b-chat') {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'chat_stream',
        model: model,
        messages: messages
      }));
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// Usage example
const wsClient = new AIHubWebSocketClient();

wsClient.onMessage('chat_chunk', (data) => {
  // Handle streaming response chunk
  console.log('Received chunk:', data.content);
});

wsClient.onMessage('chat_complete', (data) => {
  // Handle completion
  console.log('Chat complete:', data.total_tokens);
});

await wsClient.connect();
wsClient.sendStreamingChat([
  { role: 'user', content: 'Tell me a story' }
]);
```

### **2. Middleware Integration**

```python
# middleware/ai_hub_middleware.py
import asyncio
from typing import Dict, Any
from fastapi import Request, Response
from ai_client import AIHubClient

class AIHubMiddleware:
    def __init__(self, app):
        self.app = app
        self.ai_client = None
    
    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            request = Request(scope, receive)
            
            # Add AI Hub client to request state
            if not self.ai_client:
                self.ai_client = AIHubClient()
            
            request.state.ai_hub = self.ai_client
            
            # Process AI-enhanced requests
            if request.url.path.startswith("/api/ai/"):
                await self.handle_ai_request(request, receive, send)
                return
        
        await self.app(scope, receive, send)
    
    async def handle_ai_request(self, request: Request, receive, send):
        # Custom AI request handling
        body = await request.body()
        # Process with AI Hub
        # Send enhanced response
        pass

# Usage in FastAPI
from fastapi import FastAPI
app = FastAPI()
app.add_middleware(AIHubMiddleware)
```

---

## 🎯 **Best Practices**

### **1. Error Handling**

```typescript
// error-handling.ts
export class AIHubError extends Error {
  constructor(
    message: string,
    public statusCode?: number,
    public details?: any
  ) {
    super(message);
    this.name = 'AIHubError';
  }
}

export async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  delayMs = 1000
): Promise<T> {
  let lastError: Error;
  
  for (let i = 0; i <= maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      
      if (i === maxRetries) {
        throw lastError;
      }
      
      await new Promise(resolve => setTimeout(resolve, delayMs * (i + 1)));
    }
  }
  
  throw lastError!;
}
```

### **2. Caching Strategy**

```python
# caching.py
import hashlib
import json
from typing import Optional, Any
import redis

class AIResponseCache:
    def __init__(self, redis_client: redis.Redis, ttl: int = 3600):
        self.redis = redis_client
        self.ttl = ttl
    
    def _generate_key(self, messages: list, model: str) -> str:
        content = json.dumps({"messages": messages, "model": model}, sort_keys=True)
        return f"ai_response:{hashlib.md5(content.encode()).hexdigest()}"
    
    async def get(self, messages: list, model: str) -> Optional[str]:
        key = self._generate_key(messages, model)
        cached = await self.redis.get(key)
        return cached.decode() if cached else None
    
    async def set(self, messages: list, model: str, response: str):
        key = self._generate_key(messages, model)
        await self.redis.setex(key, self.ttl, response)
```

**🎉 Your AI Backend Hub is now ready for seamless integration với any application stack!** 🚀
