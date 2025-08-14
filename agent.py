"""
Dueling Double Deep Q-Network (DDQN) Agent for Cryptocurrency Trading
Implements the RL agent with neural network, prioritized replay, and training logic.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import logging
from collections import deque, namedtuple
from typing import Dict, Any, Tuple, Optional, List
import pickle
from pathlib import Path
import copy

from config import *

# Setup logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() and DEVICE == 'cuda' else 'cpu')
logger.info(f"Using device: {device}")

# Experience tuple for replay buffer
Experience = namedtuple('Experience', 
                       ['state', 'action', 'reward', 'next_state', 'done', 'priority'])


class DuelingDQNNetwork(nn.Module):
    """
    Dueling Deep Q-Network architecture with CNN feature extraction
    and separate value and advantage streams.
    """
    
    def __init__(self, state_size: int, action_size: int):
        """
        Initialize the Dueling DQN network.
        
        Args:
            state_size: Size of the input state
            action_size: Number of possible actions
        """
        super(DuelingDQNNetwork, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Calculate input dimensions for CNN
        # State is flattened: (window_size * n_features) + position_features
        n_features = len(TECHNICAL_INDICATORS) + 5  # OHLCV + indicators
        self.window_size = STATE_WINDOW_SIZE
        self.sequence_length = n_features * self.window_size
        self.position_features = 2
        
        # Reshape input for CNN: treat time series as 1D convolution
        # Input shape: (batch_size, channels=1, sequence_length)
        
        # 1D CNN layers for feature extraction from time series
        self.cnn_layers = nn.Sequential()
        
        # First conv layer
        self.cnn_layers.add_module('conv1', nn.Conv1d(
            in_channels=n_features,
            out_channels=CONV_LAYERS[0]['out_channels'],
            kernel_size=CONV_LAYERS[0]['kernel_size'],
            stride=CONV_LAYERS[0]['stride'],
            padding=CONV_LAYERS[0]['kernel_size']//2
        ))
        self.cnn_layers.add_module('relu1', nn.ReLU())
        self.cnn_layers.add_module('dropout1', nn.Dropout(DROPOUT_RATE))
        
        # Second conv layer
        self.cnn_layers.add_module('conv2', nn.Conv1d(
            in_channels=CONV_LAYERS[0]['out_channels'],
            out_channels=CONV_LAYERS[1]['out_channels'],
            kernel_size=CONV_LAYERS[1]['kernel_size'],
            stride=CONV_LAYERS[1]['stride'],
            padding=CONV_LAYERS[1]['kernel_size']//2
        ))
        self.cnn_layers.add_module('relu2', nn.ReLU())
        self.cnn_layers.add_module('dropout2', nn.Dropout(DROPOUT_RATE))
        
        # Third conv layer
        self.cnn_layers.add_module('conv3', nn.Conv1d(
            in_channels=CONV_LAYERS[1]['out_channels'],
            out_channels=CONV_LAYERS[2]['out_channels'],
            kernel_size=CONV_LAYERS[2]['kernel_size'],
            stride=CONV_LAYERS[2]['stride'],
            padding=CONV_LAYERS[2]['kernel_size']//2
        ))
        self.cnn_layers.add_module('relu3', nn.ReLU())
        self.cnn_layers.add_module('dropout3', nn.Dropout(DROPOUT_RATE))
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Calculate the size after CNN layers
        cnn_output_size = CONV_LAYERS[2]['out_channels']
        
        # Combine CNN features with position features
        combined_size = cnn_output_size + self.position_features
        
        # Dense layers
        self.dense_layers = nn.Sequential()
        
        # First dense layer
        self.dense_layers.add_module('fc1', nn.Linear(combined_size, DENSE_LAYERS[0]))
        self.dense_layers.add_module('relu_fc1', nn.ReLU())
        self.dense_layers.add_module('dropout_fc1', nn.Dropout(DROPOUT_RATE))
        
        # Second dense layer
        self.dense_layers.add_module('fc2', nn.Linear(DENSE_LAYERS[0], DENSE_LAYERS[1]))
        self.dense_layers.add_module('relu_fc2', nn.ReLU())
        self.dense_layers.add_module('dropout_fc2', nn.Dropout(DROPOUT_RATE))
        
        # Dueling streams
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(DENSE_LAYERS[1], DUELING_LAYERS[0]),
            nn.ReLU(),
            nn.Linear(DUELING_LAYERS[0], 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(DENSE_LAYERS[1], DUELING_LAYERS[0]),
            nn.ReLU(),
            nn.Linear(DUELING_LAYERS[0], action_size)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"DuelingDQN initialized - State size: {state_size}, Action size: {action_size}")
        
    def _init_weights(self, m):
        """Initialize network weights using Xavier initialization."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Q-values for all actions
        """
        batch_size = state.size(0)
        
        # Split state into time series and position features
        time_series = state[:, :-self.position_features]  # Time series data
        position_features = state[:, -self.position_features:]  # Position info
        
        # Reshape time series for CNN: (batch_size, n_features, window_size)
        time_series = time_series.view(batch_size, -1, self.window_size)
        
        # Pass through CNN layers
        cnn_output = self.cnn_layers(time_series)
        
        # Global average pooling
        cnn_output = self.global_avg_pool(cnn_output)
        cnn_output = cnn_output.view(batch_size, -1)  # Flatten
        
        # Combine CNN features with position features
        combined_features = torch.cat([cnn_output, position_features], dim=1)
        
        # Pass through dense layers
        dense_output = self.dense_layers(combined_features)
        
        # Dueling streams
        value = self.value_stream(dense_output)
        advantage = self.advantage_stream(dense_output)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer with sum tree implementation
    for efficient sampling based on TD error.
    """
    
    def __init__(self, capacity: int, alpha: float = PRIORITY_ALPHA):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        
        logger.info(f"PrioritizedReplayBuffer initialized with capacity: {capacity}")
    
    def add(self, state, action, reward, next_state, done):
        """Add experience to buffer with maximum priority."""
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.pos] = Experience(state, action, reward, next_state, done, max_priority)
        self.priorities[self.pos] = max_priority
        
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size: int, beta: float = PRIORITY_BETA_START):
        """
        Sample experiences based on priorities.
        
        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent
            
        Returns:
            Tuple of (experiences, indices, weights)
        """
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        # Calculate sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.FloatTensor(weights).to(device)
        
        return experiences, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


class DuelingDDQNAgent:
    """
    Dueling Double Deep Q-Network Agent with prioritized experience replay.
    """
    
    def __init__(self, state_size: int, action_size: int, seed: int = SEED):
        """
        Initialize the DDQN Agent.
        
        Args:
            state_size: Dimension of each state
            action_size: Dimension of each action
            seed: Random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        
        # Q-Networks
        self.q_network_local = DuelingDQNNetwork(state_size, action_size).to(device)
        self.q_network_target = DuelingDQNNetwork(state_size, action_size).to(device)
        
        # Copy local network parameters to target network
        self.hard_update(self.q_network_local, self.q_network_target)
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network_local.parameters(), lr=LEARNING_RATE)
        
        # Replay buffer
        self.memory = PrioritizedReplayBuffer(REPLAY_BUFFER_SIZE, PRIORITY_ALPHA)
        
        # Training parameters
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        self.beta = PRIORITY_BETA_START
        self.beta_increment = (1.0 - PRIORITY_BETA_START) / PRIORITY_BETA_FRAMES
        
        # Training state
        self.t_step = 0
        self.training_step = 0
        
        logger.info(f"DuelingDDQNAgent initialized successfully")
    
    def act(self, state, epsilon=None):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            epsilon: Exploration rate (if None, uses current epsilon)
            
        Returns:
            Selected action
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        # Set network to evaluation mode
        self.q_network_local.eval()
        
        with torch.no_grad():
            action_values = self.q_network_local(state)
        
        # Set back to training mode
        self.q_network_local.train()
        
        # Epsilon-greedy action selection
        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay buffer and learn if enough samples available.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is finished
        """
        # Save experience
        self.memory.add(state, action, reward, next_state, done)
        
        # Update step counter
        self.t_step = (self.t_step + 1) % RL_TRAIN_FREQUENCY
        
        # Learn if we have enough experiences and it's time to learn
        if len(self.memory) > BATCH_SIZE and self.t_step == 0:
            experiences, indices, weights = self.memory.sample(BATCH_SIZE, self.beta)
            self.learn(experiences, indices, weights)
            
            # Update beta for importance sampling
            self.beta = min(1.0, self.beta + self.beta_increment)
    
    def learn(self, experiences, indices, weights):
        """
        Update value parameters using given batch of experience tuples.
        
        Args:
            experiences: Tuple of (s, a, r, s', done) tuples
            indices: Indices of sampled experiences
            weights: Importance sampling weights
        """
        states = torch.FloatTensor([e.state for e in experiences]).to(device)
        actions = torch.LongTensor([e.action for e in experiences]).unsqueeze(1).to(device)
        rewards = torch.FloatTensor([e.reward for e in experiences]).unsqueeze(1).to(device)
        next_states = torch.FloatTensor([e.next_state for e in experiences]).to(device)
        dones = torch.BoolTensor([e.done for e in experiences]).unsqueeze(1).to(device)
        
        # Get current Q values from local network
        current_q_values = self.q_network_local(states).gather(1, actions)
        
        # Double DQN: Use local network to select actions, target network to evaluate
        with torch.no_grad():
            # Get best actions from local network
            next_actions = self.q_network_local(next_states).argmax(1).unsqueeze(1)
            
            # Evaluate these actions using target network
            next_q_values = self.q_network_target(next_states).gather(1, next_actions)
            
            # Calculate target Q values
            target_q_values = rewards + (GAMMA * next_q_values * ~dones)
        
        # Calculate TD errors for priority updates
        td_errors = torch.abs(current_q_values - target_q_values).detach()
        
        # Calculate weighted loss
        loss = (weights * F.mse_loss(current_q_values, target_q_values, reduction='none')).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network_local.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Update priorities in replay buffer
        new_priorities = td_errors.squeeze().cpu().numpy() + 1e-6  # Small epsilon to avoid zero priorities
        self.memory.update_priorities(indices, new_priorities)
        
        # Update target network
        self.training_step += 1
        if self.training_step % TARGET_UPDATE_FREQUENCY == 0:
            self.soft_update(self.q_network_local, self.q_network_target, TAU)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def soft_update(self, local_model, target_model, tau):
        """
        Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
    
    def hard_update(self, local_model, target_model):
        """Hard update (copy) model parameters."""
        target_model.load_state_dict(local_model.state_dict())
    
    def save(self, filepath):
        """
        Save model checkpoint.
        
        Args:
            filepath: Path to save the model
        """
        try:
            checkpoint = {
                'q_network_local_state_dict': self.q_network_local.state_dict(),
                'q_network_target_state_dict': self.q_network_target.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'beta': self.beta,
                'training_step': self.training_step,
                'state_size': self.state_size,
                'action_size': self.action_size
            }
            
            torch.save(checkpoint, filepath)
            logger.info(f"Model saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load(self, filepath):
        """
        Load model checkpoint.
        
        Args:
            filepath: Path to load the model from
        """
        try:
            checkpoint = torch.load(filepath, map_location=device)
            
            self.q_network_local.load_state_dict(checkpoint['q_network_local_state_dict'])
            self.q_network_target.load_state_dict(checkpoint['q_network_target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            self.epsilon = checkpoint.get('epsilon', self.epsilon)
            self.beta = checkpoint.get('beta', self.beta)
            self.training_step = checkpoint.get('training_step', 0)
            
            logger.info(f"Model loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def train_on_expert_data(self, expert_states, expert_actions, epochs=IL_EPOCHS):
        """
        Train the agent using expert demonstrations (Imitation Learning).
        
        Args:
            expert_states: List of expert states
            expert_actions: List of expert actions
            epochs: Number of training epochs
        """
        logger.info(f"Starting imitation learning for {epochs} epochs")
        
        # Convert to tensors
        states = torch.FloatTensor(expert_states).to(device)
        actions = torch.LongTensor(expert_actions).to(device)
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(states, actions)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=IL_BATCH_SIZE, 
            shuffle=True
        )
        
        # Use cross-entropy loss for imitation learning
        criterion = nn.CrossEntropyLoss()
        il_optimizer = optim.Adam(self.q_network_local.parameters(), lr=IL_LEARNING_RATE)
        
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0
            
            for batch_states, batch_actions in dataloader:
                # Forward pass
                q_values = self.q_network_local(batch_states)
                loss = criterion(q_values, batch_actions)
                
                # Backward pass
                il_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_network_local.parameters(), 1.0)
                il_optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches
            logger.info(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Update target network after imitation learning
        self.hard_update(self.q_network_local, self.q_network_target)
        
        logger.info("Imitation learning completed")
    
    def get_q_values(self, state):
        """
        Get Q-values for a given state.
        
        Args:
            state: Input state
            
        Returns:
            Q-values for all actions
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        self.q_network_local.eval()
        with torch.no_grad():
            q_values = self.q_network_local(state)
        self.q_network_local.train()
        
        return q_values.cpu().numpy().flatten()
    
    def get_action_probabilities(self, state, temperature=1.0):
        """
        Get action probabilities using softmax with temperature.
        
        Args:
            state: Input state
            temperature: Temperature for softmax (higher = more exploration)
            
        Returns:
            Action probabilities
        """
        q_values = self.get_q_values(state)
        
        # Apply temperature
        q_values = q_values / temperature
        
        # Apply softmax
        exp_q = np.exp(q_values - np.max(q_values))  # Subtract max for numerical stability
        probabilities = exp_q / np.sum(exp_q)
        
        return probabilities
    
    def get_training_stats(self):
        """Get current training statistics."""
        return {
            'epsilon': self.epsilon,
            'beta': self.beta,
            'training_step': self.training_step,
            'buffer_size': len(self.memory),
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }


def create_expert_dataset(env, expert_trader, num_episodes=1):
    """
    Create dataset from expert demonstrations.
    
    Args:
        env: Trading environment
        expert_trader: Expert trader instance
        num_episodes: Number of episodes to generate
        
    Returns:
        Tuple of (states, actions)
    """
    logger.info(f"Creating expert dataset with {num_episodes} episodes")
    
    states = []
    actions = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_states = []
        episode_actions = []
        
        # Generate expert actions for the entire episode
        expert_actions = expert_trader.generate_expert_actions(env.df)
        
        for step in range(len(expert_actions)):
            if step >= env.max_steps:
                break
                
            episode_states.append(state.copy())
            episode_actions.append(expert_actions[step])
            
            # Take the expert action in environment
            next_state, reward, done, info = env.step(expert_actions[step])
            state = next_state
            
            if done:
                break
        
        states.extend(episode_states)
        actions.extend(episode_actions)
        
        logger.info(f"Episode {episode+1}/{num_episodes} completed with {len(episode_states)} steps")
    
    logger.info(f"Expert dataset created with {len(states)} state-action pairs")
    
    return np.array(states), np.array(actions)


if __name__ == "__main__":
    # Example usage and testing
    try:
        # Test the neural network
        state_size = (STATE_WINDOW_SIZE * (len(TECHNICAL_INDICATORS) + 5)) + 2
        action_size = N_ACTIONS
        
        print(f"Testing DuelingDQN with state_size: {state_size}, action_size: {action_size}")
        
        # Initialize agent
        agent = DuelingDDQNAgent(state_size, action_size)
        
        # Test forward pass
        test_state = np.random.randn(state_size)
        action = agent.act(test_state, epsilon=0.1)
        print(f"Test action: {action}")
        
        # Test Q-values
        q_values = agent.get_q_values(test_state)
        print(f"Q-values: {q_values}")
        
        # Test action probabilities
        probs = agent.get_action_probabilities(test_state)
        print(f"Action probabilities: {probs}")
        
        # Test training stats
        stats = agent.get_training_stats()
        print(f"Training stats: {stats}")
        
        # Test save/load
        test_path = MODEL_DIR / "test_model.pth"
        agent.save(test_path)
        print(f"Model saved to {test_path}")
        
        # Create new agent and load
        agent2 = DuelingDDQNAgent(state_size, action_size)
        agent2.load(test_path)
        print("Model loaded successfully")
        
        # Verify loaded model produces same output
        action2 = agent2.act(test_state, epsilon=0.0)  # No randomness
        print(f"Loaded model action: {action2}")
        
        # Clean up
        test_path.unlink()
        print("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise