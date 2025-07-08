# Claude ↔ ChatGPT Bridge

*A production-ready AI bridge enabling seamless conversations between Anthropic Claude and OpenAI ChatGPT with comprehensive error handling, retry logic, and dual interfaces.*

## 🎯 Project Overview

**Development Timeline**: 8+ weeks (May 2025 - July 2025)  
**Architecture**: Full-stack AI integration with CLI and web interfaces  
**Testing**: 9/9 comprehensive test suite with CI/CD integration  

This project demonstrates enterprise-level AI system integration with robust error handling, production deployment readiness, and professional user experience design.

## ✨ Key Features

### 🔧 **Production Engineering**
- **Exponential backoff retry logic** with Fibonacci jitter
- **Comprehensive error handling** for API failures and rate limits
- **Graceful degradation** when services are overloaded
- **Environment-based configuration** with secure secrets management
- **GitHub Actions CI/CD** with automated testing and linting

### 🤖 **AI Integration**
- **Dual API support** for Claude (Anthropic) and ChatGPT (OpenAI)
- **Intelligent conversation bridging** with multi-turn support
- **Response comparison and analysis** with keyword-based evaluation
- **Token usage tracking** and cost optimization
- **Model flexibility** with configurable endpoints

### 🎨 **User Experience**
- **Streamlit web interface** with real-time side-by-side comparison
- **CLI tool** for programmatic access and automation
- **Loading states and progress indicators** for async operations
- **Helpful error messages** with recovery suggestions
- **API status monitoring** and configuration validation

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/kilday-human/claude-chat-bridge.git
cd claude-chat-bridge
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your actual API keys

# Test with mock responses (no API calls)
python cli_bridge.py "Compare AI safety approaches" 2 --mock

# Launch web interface
streamlit run bridge.py
```

## 🏗️ Architecture

### **Project Structure**
```
claude-chat-bridge/
├── 🎯 Core Bridge
│   ├── cli_bridge.py          # CLI interface with conversation management
│   ├── bridge.py              # Streamlit web application
│   └── backoff_utils.py       # Fibonacci jitter retry logic
├── 🔌 API Wrappers
│   ├── claude_wrapper.py      # Anthropic Claude integration
│   ├── chatgpt_wrapper.py     # OpenAI ChatGPT integration
│   └── metrics.py             # Response analysis and comparison
├── 🧪 Testing Infrastructure
│   ├── tests/                 # Comprehensive test suite (9 tests)
│   ├── .github/workflows/     # CI/CD automation
│   └── requirements.txt       # Dependency management
└── 📚 Documentation
    ├── README.md              # This file
    └── .env.example           # Configuration template
```

### **Technical Stack**
- **Languages**: Python 3.13+
- **AI APIs**: Anthropic Claude, OpenAI GPT
- **Web Framework**: Streamlit
- **Testing**: pytest with comprehensive coverage
- **CI/CD**: GitHub Actions
- **Infrastructure**: Environment-based config, Docker-ready

## 💻 Usage Examples

### **CLI Mode**
```bash
# Multi-turn conversation bridge
python cli_bridge.py "Explain quantum computing" 3

# Mock mode for testing (no API costs)
python cli_bridge.py "Write a haiku about AI" 1 --mock
```

### **Web Interface**
```bash
streamlit run bridge.py
```
- Enter prompts in the web interface
- Compare responses side-by-side
- Analyze differences with keyword highlighting
- Graceful error handling for API failures

### **Programmatic Usage**
```python
from claude_wrapper import send_to_claude
from chatgpt_wrapper import send_to_chatgpt

messages = [{"role": "user", "content": "Hello!"}]
claude_response = send_to_claude(messages)
gpt_response = send_to_chatgpt(messages)
```

## 🧪 Testing & Quality Assurance

```bash
# Run comprehensive test suite
pytest -v

# All 9 tests covering:
# ✅ API wrapper functionality
# ✅ Conversation bridge logic  
# ✅ Error handling and retry mechanisms
# ✅ Latency monitoring and metrics
# ✅ Backoff utility functions
```

**Test Coverage Areas**:
- Basic wrapper functionality for both APIs
- Multi-turn conversation handling
- Error scenarios and retry logic
- Rate limiting and overload conditions
- Metrics and response analysis

## 🔒 Security & Configuration

### **Environment Variables**
```bash
# Required API keys
OPENAI_API_KEY=your-openai-api-key-here
CLAUDE_API_KEY=your-anthropic-api-key-here
```

### **Security Features**
- ✅ Environment variable-based secrets management
- ✅ API key validation and sanitization
- ✅ Secure headers and request handling
- ✅ No hardcoded credentials in codebase
- ✅ .gitignore protection for sensitive files

## 🎛️ Advanced Features

### **Error Handling & Resilience**
- **Automatic retries** with exponential backoff
- **Overload detection** and graceful fallback
- **Rate limit awareness** with intelligent delays
- **Timeout handling** and connection management
- **Detailed logging** for debugging and monitoring

### **Response Analysis**
- **Coverage comparison** between AI models
- **Length parity analysis** for response balance
- **Safety filtering** for content validation
- **Keyword-based evaluation** for targeted analysis
- **JSON metrics export** for further processing

## 🚀 Deployment Ready

### **Production Features**
- **Environment-based configuration** for different deployment stages
- **Comprehensive error handling** for production stability
- **Monitoring and logging** capabilities
- **Resource optimization** with connection pooling
- **Scalable architecture** supporting multiple concurrent users

### **CI/CD Pipeline**
- **Automated testing** on every commit
- **Code quality checks** with linting
- **Dependency scanning** for security
- **Multi-environment support** (dev/staging/prod)

## 🔄 Recent Updates

- **v2.1** (July 2025): Added robust Streamlit interface with retry logic
- **v2.0** (July 2025): Comprehensive error handling and production polish
- **v1.5** (June 2025): Enhanced testing suite and CI/CD integration
- **v1.0** (May 2025): Initial release with core bridge functionality

## 📈 Engineering Highlights

This project showcases **senior-level engineering practices**:

- **8+ week sustained development** demonstrating project ownership
- **Production engineering mindset** with comprehensive error handling
- **Full-stack capabilities** from CLI tools to web interfaces
- **API integration expertise** with multiple external services
- **Testing discipline** with comprehensive coverage
- **User experience focus** with polished interfaces and helpful guidance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Requirements**: All new features must include tests and maintain the existing test coverage.

## 📄 License

MIT License - feel free to use this project for your own AI experiments and production deployments!

---

*Built with ❤️ for the AI community. Demonstrating production-ready AI system integration with enterprise-level reliability.*
