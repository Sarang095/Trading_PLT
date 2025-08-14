# Cryptocurrency Trading Bot with Dueling DDQN and Imitation Learning

A sophisticated cryptocurrency trading bot that uses a hybrid approach combining **Dueling Double Deep Q-Network (DDQN)** and **Imitation Learning (IL)** to trade BTC/USDT based on market data and sentiment analysis.

## 🌟 Key Features

- **Hybrid AI Approach**: Combines Imitation Learning from expert strategies with Reinforcement Learning
- **Advanced Architecture**: Dueling DDQN with Prioritized Experience Replay
- **Multi-Modal Analysis**: Technical indicators + sentiment analysis from financial news
- **Expert Strategy**: EMA + Heikin Ashi + Parabolic SAR trading strategy
- **Real-Time Trading**: Live WebSocket data feeds with automatic reconnection
- **Risk Management**: Built-in safety checks, position limits, and drawdown protection
- **Comprehensive Monitoring**: Detailed logging, performance metrics, and visualization
- **Production Ready**: Modular design, error handling, and configuration management

## 🚀 How Our Platform Works

### 1. **Two-Phase Training Approach**

#### Phase 1: Imitation Learning (Behavioral Cloning)
- Expert trader generates optimal actions using technical analysis
- Neural network learns to mimic expert behavior through supervised learning
- Creates a strong baseline policy before reinforcement learning

#### Phase 2: Reinforcement Learning Fine-tuning
- Agent improves upon expert strategy through environment interaction
- Uses Dueling DDQN with prioritized experience replay
- Multi-objective reward function optimizing for profit, Sharpe ratio, and risk

### 2. **Technical Architecture**

```
Data Sources → Feature Engineering → State Representation → Agent Decision → Action Execution
     ↓              ↓                      ↓                  ↓              ↓
- Market Data   - Technical         - Normalized          - DDQN        - Buy/Sell/Hold
- News/Social   - Indicators        - Time Series         - Network     - Risk Checks
- Sentiment     - Sentiment         - Position Info       - Output      - Order Execution
```

### 3. **Expert Strategy Components**

- **EMA Trend Analysis**: 9, 21, and 200-period exponential moving averages
- **Heikin Ashi Candles**: Smoothed price action for trend identification
- **Parabolic SAR**: Dynamic stop-loss and trend reversal detection
- **Multi-Condition Signals**: Combines all indicators for robust decision making

### 4. **Neural Network Architecture**

- **Input Layer**: Normalized market data + technical indicators + position info
- **CNN Layers**: 1D convolutions for time-series feature extraction
- **Dueling Streams**: Separate value and advantage estimation
- **Output Layer**: Q-values for Hold/Buy/Sell actions

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster training)
- At least 8GB RAM
- Stable internet connection for live trading

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/crypto-trading-bot.git
cd crypto-trading-bot
```

### 2. Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n crypto-bot python=3.9
conda activate crypto-bot

# Or using venv
python -m venv crypto-bot-env
source crypto-bot-env/bin/activate  # On Windows: crypto-bot-env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install TA-Lib (Technical Analysis Library)

#### On Ubuntu/Debian:
```bash
sudo apt-get install ta-lib
```

#### On macOS:
```bash
brew install ta-lib
```

#### On Windows:
Download the appropriate .whl file from [TA-Lib releases](https://github.com/mrjbq7/ta-lib) and install:
```bash
pip install TA_Lib-0.4.25-cp39-cp39-win_amd64.whl  # Adjust for your Python version
```

### 5. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# Exchange API Keys (for live trading)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

# Optional: Other API keys
NEWS_API_KEY=your_news_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
```

## 🎯 Quick Start Guide

### Step 1: Validate Configuration

```bash
python config.py
```

This will validate all configuration parameters and create necessary directories.

### Step 2: Test Individual Components

Test data fetching:
```bash
python data_manager.py
```

Test expert strategy:
```bash
python expert.py
```

Test trading environment:
```bash
python trading_env.py
```

Test agent:
```bash
python agent.py
```

### Step 3: Train the Model

#### Option A: Complete Training Pipeline
```bash
python train.py
```

#### Option B: Partial Training
```bash
# Only Imitation Learning
python train.py --skip-rl

# Only Reinforcement Learning (requires pre-trained model)
python train.py --skip-il --load-model models/expert_pretrained.pth
```

#### Monitor Training Progress
Training logs and metrics are saved in:
- `logs/training.log` - Training logs
- `results/training_report_*/` - Performance plots and reports

### Step 4: Run Live Trading

#### Paper Trading (Recommended for testing)
```bash
python trade.py --testnet
```

#### Live Trading (Use with caution)
```bash
python trade.py --live --model models/final_model.pth
```

## 📊 Understanding the Output

### Training Metrics

During training, you'll see outputs like:
```
Episode    100 | Timesteps:     2500 | Avg Reward:    15.23 | Avg Length:   85.1 | Epsilon: 0.950 | Buffer:   2500
Episode    200 | Timesteps:     5000 | Avg Reward:    28.45 | Avg Length:   89.3 | Epsilon: 0.903 | Buffer:   5000
```

- **Episode**: Training episode number
- **Timesteps**: Total environment steps
- **Avg Reward**: Average reward over last 100 episodes
- **Avg Length**: Average episode length
- **Epsilon**: Exploration rate (decreases over time)
- **Buffer**: Number of experiences in replay buffer

### Live Trading Output

```
Status: Price=$43,250.50, Position=HOLD, Daily P&L=$125.30, Total Trades=15
Buy order executed: 0.002315 BTC at $43,250.50
Sell order executed: 0.002315 BTC at $43,780.20
Realized P&L: $1.23 (1.22%)
```

## 📁 Project Structure

```
crypto-trading-bot/
├── config.py              # Configuration and hyperparameters
├── data_manager.py         # Data fetching and WebSocket handling
├── sentiment_analyzer.py   # News sentiment analysis using FinBERT
├── expert.py              # Expert trading strategy implementation
├── trading_env.py          # Custom Gym environment for RL training
├── agent.py               # Dueling DDQN agent with prioritized replay
├── train.py               # Two-phase training pipeline
├── trade.py               # Live trading execution script
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── data/                 # Market data and trading logs
├── models/               # Trained model checkpoints
├── logs/                 # Application logs
└── results/              # Training reports and visualizations
```

## ⚙️ Configuration

### Key Configuration Parameters

Edit `config.py` to customize:

```python
# Trading Configuration
CRYPTO_PAIR = 'BTC/USDT'
INITIAL_BALANCE = 10000
TRANSACTION_FEE_PERCENT = 0.001

# Model Configuration
STATE_WINDOW_SIZE = 60
LEARNING_RATE = 0.0001
GAMMA = 0.99

# Training Configuration
IL_EPOCHS = 10
RL_TOTAL_TIMESTEPS = 1000000

# Risk Management
MAX_POSITION_SIZE = 0.95
STOP_LOSS_PERCENT = 0.05
MAX_DRAWDOWN_PERCENT = 0.20
```

### Environment Variables

For production use, set these environment variables:

```bash
export BINANCE_API_KEY="your_api_key"
export BINANCE_SECRET_KEY="your_secret_key"
export ENVIRONMENT="production"  # or "development"
```

## 🔧 Advanced Usage

### Custom Expert Strategies

Modify the `ExpertTrader` class in `expert.py` to implement your own trading strategy:

```python
def generate_expert_actions(self, df):
    # Your custom strategy logic here
    actions = []
    for i, row in df.iterrows():
        if your_buy_condition:
            actions.append(1)  # Buy
        elif your_sell_condition:
            actions.append(2)  # Sell
        else:
            actions.append(0)  # Hold
    return actions
```

### Custom Reward Functions

Modify the reward function in `trading_env.py`:

```python
def _calculate_reward(self, action, executed, prev_net_worth, prev_position):
    # Your custom reward logic
    reward = your_profit_component + your_risk_component
    return reward
```

### Adding New Technical Indicators

Add indicators in `data_manager.py`:

```python
def _calculate_technical_indicators(self, df):
    # Existing indicators...
    
    # Add your custom indicator
    df['your_indicator'] = calculate_your_indicator(df)
    return df
```

Then update `TECHNICAL_INDICATORS` in `config.py`.

### Hyperparameter Tuning

Use tools like Optuna for automated hyperparameter optimization:

```python
import optuna

def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # Train model with suggested parameters
    # Return validation performance
    return validation_score

study = optuna.create_study()
study.optimize(objective, n_trials=100)
```

## 📈 Performance Monitoring

### Key Metrics to Monitor

1. **Total Return**: Overall portfolio performance
2. **Sharpe Ratio**: Risk-adjusted returns
3. **Maximum Drawdown**: Worst peak-to-trough decline
4. **Win Rate**: Percentage of profitable trades
5. **Average Trade Duration**: How long positions are held

### Visualization Tools

The training script automatically generates:
- Episode reward progression
- Performance metric trends
- Action distribution analysis
- Profit/loss distribution

### Real-time Monitoring

For production deployments, consider integrating:
- **Grafana + InfluxDB**: Time-series monitoring
- **Prometheus**: Metrics collection
- **Slack/Discord**: Trade notifications
- **TradingView**: Chart analysis

## 🛡️ Risk Management

### Built-in Safety Features

1. **Position Limits**: Maximum position size relative to portfolio
2. **Daily Trade Limits**: Maximum number of trades per day
3. **Loss Limits**: Daily and total loss thresholds
4. **Data Quality Checks**: Stale data detection
5. **Connection Monitoring**: WebSocket health checks

### Recommended Safety Practices

1. **Start with Paper Trading**: Always test with simulated money first
2. **Gradual Capital Allocation**: Start small and increase gradually
3. **Regular Monitoring**: Check bot performance frequently
4. **Stop-Loss Orders**: Implement additional safety nets
5. **Diversification**: Don't put all funds in one bot

### Emergency Procedures

If something goes wrong:

1. **Stop the Bot**: Press Ctrl+C or kill the process
2. **Check Positions**: Verify current holdings on exchange
3. **Manual Override**: Close positions manually if needed
4. **Review Logs**: Analyze what happened
5. **Adjust Configuration**: Fix issues before restarting

## 🔍 Troubleshooting

### Common Issues

#### 1. **Import Errors**
```bash
ModuleNotFoundError: No module named 'talib'
```
**Solution**: Install TA-Lib following the installation guide above.

#### 2. **CUDA Out of Memory**
```bash
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size in `config.py` or use CPU training:
```python
DEVICE = 'cpu'
BATCH_SIZE = 32  # Reduce from 64
```

#### 3. **WebSocket Connection Issues**
```bash
WebSocket connection failed
```
**Solution**: Check internet connection and API credentials. Try using testnet.

#### 4. **Invalid API Credentials**
```bash
ccxt.AuthenticationError: Invalid API key
```
**Solution**: Verify API keys in `.env` file and ensure they have trading permissions.

#### 5. **Insufficient Training Data**
```bash
ValueError: Insufficient data: 500 < 1000
```
**Solution**: Increase `HISTORICAL_DATA_LOOKBACK_DAYS` in config or reduce `MIN_DATA_POINTS`.

### Debug Mode

Enable detailed logging:

```python
# In config.py
LOG_LEVEL = 'DEBUG'
```

### Getting Help

1. Check the logs in `logs/` directory
2. Review configuration in `config.py`
3. Test individual components separately
4. Reduce complexity (smaller models, less data)
5. Use CPU-only mode for debugging

## 🚨 Important Disclaimers

### Financial Risk Warning

⚠️ **HIGH RISK WARNING**: Cryptocurrency trading involves substantial risk of loss. This bot:

- Is for educational and research purposes
- May lose money in live trading
- Should not be used with funds you cannot afford to lose
- Does not guarantee profits
- Past performance does not predict future results

### Legal Considerations

- Ensure compliance with local regulations
- Some jurisdictions restrict algorithmic trading
- Tax implications may apply to trading profits
- Consider consulting with financial advisors

### Technical Limitations

- Model performance depends on market conditions
- Requires significant computational resources
- Network connectivity issues can affect performance
- No guarantee of execution in volatile markets

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure code follows PEP 8 style
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pre-commit black flake8 pytest

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI Gym**: Reinforcement learning environments
- **PyTorch**: Deep learning framework
- **CCXT**: Cryptocurrency exchange integration
- **Hugging Face**: Pre-trained NLP models
- **TA-Lib**: Technical analysis indicators

## 📞 Support

For questions, issues, or contributions:

- 📧 Email: support@crypto-trading-bot.com
- 💬 Discord: [Join our community](https://discord.gg/crypto-trading-bot)
- 🐛 Issues: [GitHub Issues](https://github.com/your-username/crypto-trading-bot/issues)
- 📚 Documentation: [Wiki](https://github.com/your-username/crypto-trading-bot/wiki)

---

**Happy Trading! 🚀**

*Remember: Only trade with money you can afford to lose, and always do your own research.*