# ai_predictor.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from config import API_KEY, BASE_URL

class AdvancedForexPredictor:
    def __init__(self, lookback_period=50, regression_window=20):
        self.model = None
        self.feature_columns = []
        self.is_trained = False
        self.lookback_period = lookback_period
        self.regression_window = regression_window
        
    def calculate_confidence_tier(self, confidence):
        if confidence >= 0.7:
            return "high", 1, 50
        elif confidence >= 0.6:
            return "medium", 3, 30
        elif confidence >= 0.5:
            return "low", 5, 20
        else:
            return "very_low", 0, 0
    
    def detect_three_line_strike(self, df):
        """
        Detect Three Line Strike candlestick pattern with improved accuracy
        """
        if len(df) < 4:
            return 0
            
        # Get last 4 candles
        opens = df['open'].values[-4:]
        highs = df['high'].values[-4:]
        lows = df['low'].values[-4:]
        closes = df['close'].values[-4:]
        
        # Bullish Three Line Strike - More strict conditions
        if (closes[0] < opens[0] and  # First candle bearish
            closes[1] < opens[1] and  # Second candle bearish  
            closes[2] < opens[2] and  # Third candle bearish
            closes[3] > opens[3] and  # Fourth candle bullish
            closes[3] > opens[0] and  # Closes above first open
            lows[3] <= lows[0:3].min() and  # Tests previous lows
            # Additional confirmation: fourth candle should have strong body
            (closes[3] - opens[3]) / (highs[3] - lows[3]) > 0.6):  # Strong bullish body
            return 1
            
        # Bearish Three Line Strike - More strict conditions
        elif (closes[0] > opens[0] and  # First candle bullish
              closes[1] > opens[1] and  # Second candle bullish
              closes[2] > opens[2] and  # Third candle bullish
              closes[3] < opens[3] and  # Fourth candle bearish
              closes[3] < opens[0] and  # Closes below first open
              highs[3] >= highs[0:3].max() and  # Tests previous highs
              # Additional confirmation: fourth candle should have strong body
              (opens[3] - closes[3]) / (highs[3] - lows[3]) > 0.6):  # Strong bearish body
            return -1
        else:
            return 0
    
    def calculate_support_resistance(self, df, window=20):
        """
        Identify dynamic support and resistance levels with improved accuracy
        """
        if len(df) < window:
            return {'support': 0, 'resistance': 0, 'pivot': 0}
            
        # Use multiple methods for better accuracy
        recent_highs = df['high'].rolling(window=window).max()
        recent_lows = df['low'].rolling(window=window).min()
        
        # Fibonacci retracement levels for additional confirmation
        recent_high = df['high'].iloc[-window:].max()
        recent_low = df['low'].iloc[-window:].min()
        diff = recent_high - recent_low
        
        # Key Fibonacci levels
        fib_38 = recent_high - 0.382 * diff
        fib_50 = recent_high - 0.5 * diff
        fib_62 = recent_high - 0.618 * diff
        
        resistance = max(recent_highs.iloc[-1], fib_38)
        support = min(recent_lows.iloc[-1], fib_62)
        
        # Pivot point levels
        pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
        r1 = 2 * pivot - df['low'].iloc[-1]
        s1 = 2 * pivot - df['high'].iloc[-1]
        
        return {
            'support': min(support, s1, fib_62),
            'resistance': max(resistance, r1, fib_38),
            'pivot': pivot
        }
    
    def linear_regression_trend(self, prices):
        """
        Calculate linear regression slope and strength with improved accuracy
        """
        if len(prices) < self.regression_window:
            return 0, 0
            
        x = np.arange(len(prices)).reshape(-1, 1)
        y = prices.values
        
        model = LinearRegression()
        model.fit(x, y)
        
        slope = model.coef_[0]
        y_pred = model.predict(x)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Adjust slope for price scale
        slope = slope / np.mean(y) if np.mean(y) != 0 else slope
        
        return slope, r_squared
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI using pandas only"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD using pandas only"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        return macd, macd_signal, macd_histogram
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands using pandas only"""
        middle = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = middle + (std * num_std)
        lower = middle - (std * num_std)
        return upper, lower, middle
    
    def calculate_atr(self, high, low, close, window=14):
        """Calculate ATR using pandas only"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = np.maximum(np.maximum(high_low, high_close), low_close)
        atr = true_range.rolling(window=window).mean()
        return atr
    
    def calculate_market_momentum(self, df):
        """
        Calculate comprehensive market momentum indicators using built-in functions
        """
        if len(df) < 20:
            return {'momentum': 0, 'trend_strength': 0, 'volatility': 0}
        
        # RSI momentum
        rsi = self.calculate_rsi(df['close'], 14)
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        
        # MACD momentum
        macd, macd_signal, _ = self.calculate_macd(df['close'])
        macd_value = macd_signal.iloc[-1] if not macd_signal.empty else 0
        
        # Price momentum
        price_momentum = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        
        # Volume momentum (if available)
        volume_momentum = 0
        if 'volume' in df.columns:
            volume_momentum = (df['volume'].iloc[-1] - df['volume'].rolling(5).mean().iloc[-1]) / df['volume'].rolling(5).mean().iloc[-1]
        
        # Combined momentum score
        momentum_score = (
            0.4 * ((current_rsi - 50) / 50) +  # RSI contribution
            0.3 * (macd_value * 100) +  # MACD contribution
            0.2 * (price_momentum * 100) +  # Price momentum
            0.1 * volume_momentum  # Volume momentum
        )
        
        # Trend strength based on moving average alignment
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        sma_50 = df['close'].rolling(50).mean().iloc[-1]
        trend_strength = 1 if (df['close'].iloc[-1] > sma_20 > sma_50) else -1 if (df['close'].iloc[-1] < sma_20 < sma_50) else 0
        
        return {
            'momentum': momentum_score,
            'trend_strength': trend_strength,
            'rsi': current_rsi,
            'macd': macd_value
        }
    
    def calculate_signal_score(self, df, current_price):
        """
        Calculate comprehensive signal score with improved weighting
        """
        score = 0
        max_score = 10

        # 1. Market Momentum (40% weight)
        momentum_data = self.calculate_market_momentum(df)
        momentum_score = np.tanh(momentum_data['momentum']) * 2  # Normalize to -2 to 2
        score += 2 + momentum_score  # Base 2 + momentum adjustment
        
        # 2. Three Line Strike Pattern (20% weight)
        strike_pattern = self.detect_three_line_strike(df)
        if strike_pattern != 0:
            score += 2 * strike_pattern  # +2 for bullish, -2 for bearish
        
        # 3. Support/Resistance Alignment (20% weight)
        levels = self.calculate_support_resistance(df)
        if levels:
            distance_to_support = abs(current_price - levels['support'])
            distance_to_resistance = abs(current_price - levels['resistance'])
            price_range = levels['resistance'] - levels['support']
            
            if price_range > 0:
                # Score higher when price is near support for buys, near resistance for sells
                support_proximity = 1 - (distance_to_support / price_range)
                resistance_proximity = 1 - (distance_to_resistance / price_range)
                
                # If momentum is positive and near support, boost score
                if momentum_data['momentum'] > 0 and support_proximity > 0.7:
                    score += 1
                # If momentum is negative and near resistance, boost score
                elif momentum_data['momentum'] < 0 and resistance_proximity > 0.7:
                    score += 1
        
        # 4. Regression Trend Strength (20% weight)
        closes = df['close'].iloc[-self.regression_window:]
        slope, r_squared = self.linear_regression_trend(closes)
        
        # Only count strong trends with good R-squared
        if abs(slope) > 0.0001 and r_squared > 0.3:
            trend_strength = min(r_squared * 2, 1.0) * np.sign(slope)
            score += trend_strength * 2
        
        return min(max(score, 0), max_score), max_score
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features with built-in technical indicators"""
        df = data.copy()
        
        # Basic price features
        df['price_range'] = (df['high'] - df['low']) / df['open']
        df['price_change'] = (df['close'] - df['open']) / df['open']
        df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 0.0001)
        
        # Moving averages using built-in functions
        for window in [3, 5, 8, 13, 21, 34, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
            df[f'price_vs_sma{window}'] = df['close'] / df[f'sma_{window}'].replace(0, 1)
        
        # RSI indicators
        for period in [7, 14]:
            df[f'rsi_{period}'] = self.calculate_rsi(df['close'], period)
        
        # MACD
        macd, macd_signal, macd_histogram = self.calculate_macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_histogram'] = macd_histogram
        
        # Bollinger Bands
        bb_upper, bb_lower, bb_middle = self.calculate_bollinger_bands(df['close'])
        df['bb_upper'] = bb_upper
        df['bb_lower'] = bb_lower
        df['bb_middle'] = bb_middle
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volatility features
        df['volatility'] = df['close'].pct_change().rolling(window=5).std().fillna(0)
        df['atr'] = self.calculate_atr(df['high'], df['low'], df['close'], 14)
        
        # Volume features (if available)
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=5).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma'].replace(0, 1)
            df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], 1.0)
            # Volume-based indicators - simple OBV
            df['volume_obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Linear regression features
        regression_features = []
        for i in range(len(df)):
            if i >= self.regression_window:
                window_data = df['close'].iloc[i-self.regression_window:i]
                slope, r_squared = self.linear_regression_trend(window_data)
                regression_features.append({
                    'regression_slope': slope,
                    'trend_strength': r_squared
                })
            else:
                regression_features.append({'regression_slope': 0, 'trend_strength': 0})
        
        regression_df = pd.DataFrame(regression_features, index=df.index)
        df = pd.concat([df, regression_df], axis=1)
        
        # Pattern detection
        pattern_features = []
        for i in range(len(df)):
            if i >= 4:
                window_data = df.iloc[i-4:i+1]
                three_line_strike = self.detect_three_line_strike(window_data)
                pattern_features.append({
                    'three_line_strike': three_line_strike
                })
            else:
                pattern_features.append({'three_line_strike': 0})
        
        pattern_df = pd.DataFrame(pattern_features, index=df.index)
        df = pd.concat([df, pattern_df], axis=1)
        
        # Momentum features
        momentum_features = []
        for i in range(len(df)):
            if i >= 20:
                window_data = df.iloc[i-20:i+1]
                momentum_data = self.calculate_market_momentum(window_data)
                momentum_features.append(momentum_data)
            else:
                momentum_features.append({'momentum': 0, 'trend_strength': 0, 'rsi': 50, 'macd': 0})
        
        momentum_df = pd.DataFrame(momentum_features, index=df.index)
        df = pd.concat([df, momentum_df], axis=1)
        
        return df.fillna(0)
    
    def prepare_target(self, data: pd.DataFrame, prediction_horizon: int = 3) -> pd.Series:
        """Improved target preparation with trend confirmation"""
        # Use future price movement over multiple periods
        future_price = data['close'].shift(-prediction_horizon)
        current_price = data['close']
        
        # Calculate future return
        future_return = (future_price - current_price) / current_price
        
        # Only trade if the move is significant (more than 0.1%)
        target = ((future_return > 0.001) & (future_return.abs() > 0.0005)).astype(int)
        target = target.fillna(0)
        
        return target
    
    def train_model(self, data: pd.DataFrame, timeframe: str = "1h"):
        """Improved model training with better features and parameters"""
        df_with_features = self.create_features(data)
        target = self.prepare_target(df_with_features, 3)  # 3-period prediction
        
        # Select most important features
        self.feature_columns = [
            'open', 'high', 'low', 'close', 'price_range', 'price_change', 
            'body_size', 'sma_5', 'sma_8', 'sma_13', 'sma_21', 'sma_34',
            'ema_5', 'ema_8', 'ema_13', 'ema_21',
            'price_vs_sma5', 'price_vs_sma8', 'price_vs_sma13', 'price_vs_sma21',
            'rsi_7', 'rsi_14', 'macd', 'macd_signal', 'macd_histogram',
            'bb_position', 'volatility', 'atr', 'regression_slope', 'trend_strength',
            'three_line_strike', 'momentum', 'trend_strength'
        ]
        
        if 'volume' in df_with_features.columns:
            self.feature_columns.extend(['volume', 'volume_ratio', 'volume_obv'])
        
        available_features = [col for col in self.feature_columns if col in df_with_features.columns]
        X = df_with_features[available_features]
        y = target
        
        valid_indices = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_indices]
        y_clean = y[valid_indices]
        
        if len(X_clean) < 20:
            raise ValueError("Insufficient data for training")
        
        # Improved model parameters
        self.model = RandomForestClassifier(
            n_estimators=100,  # More trees for better accuracy
            max_depth=15,      # Deeper trees for complex patterns
            min_samples_split=5,
            min_samples_leaf=3,
            max_features='sqrt',  # Better feature selection
            bootstrap=True,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_clean, y_clean)
        self.is_trained = True
        
        return self.model
    
    def predict_with_confidence(self, latest_data: pd.DataFrame):
        """Improved prediction with momentum confirmation"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        latest_with_features = self.create_features(latest_data)
        
        available_features = [col for col in self.feature_columns if col in latest_with_features.columns]
        X_new = latest_with_features[available_features].iloc[-1:]
        
        if X_new.isna().any().any():
            X_new = X_new.fillna(0)
        
        prediction = self.model.predict(X_new)[0]
        probabilities = self.model.predict_proba(X_new)[0]
        confidence = max(probabilities)
        
        # Get additional signal information
        current_price = latest_data['close'].iloc[-1]
        signal_score, max_score = self.calculate_signal_score(latest_data, current_price)
        levels = self.calculate_support_resistance(latest_data)
        momentum_data = self.calculate_market_momentum(latest_data)
        
        # Momentum-based confirmation
        momentum_confirmation = 0
        if prediction == 1 and momentum_data['momentum'] > 0:
            momentum_confirmation = 1
        elif prediction == 0 and momentum_data['momentum'] < 0:
            momentum_confirmation = 1
        
        # Adjust confidence based on momentum confirmation
        adjusted_confidence = confidence
        if momentum_confirmation:
            adjusted_confidence = min(confidence * 1.2, 0.95)  # Boost confidence
        else:
            adjusted_confidence = confidence * 0.8  # Reduce confidence
        
        # Final prediction with momentum filter
        final_prediction = prediction
        if prediction == 1 and momentum_data['momentum'] < -0.5:  # Strong negative momentum
            final_prediction = 0  # Override to sell
            adjusted_confidence = max(probabilities[0])  # Use sell probability
        elif prediction == 0 and momentum_data['momentum'] > 0.5:  # Strong positive momentum
            final_prediction = 1  # Override to buy
            adjusted_confidence = max(probabilities[1])  # Use buy probability
        
        return {
            'prediction': final_prediction,
            'confidence': min(adjusted_confidence, 0.95),
            'probabilities': probabilities,
            'signal_score': signal_score,
            'max_score': max_score,
            'support_level': levels['support'],
            'resistance_level': levels['resistance'],
            'regression_slope': latest_with_features['regression_slope'].iloc[-1],
            'three_line_strike': latest_with_features['three_line_strike'].iloc[-1],
            'momentum': momentum_data['momentum'],
            'rsi': momentum_data.get('rsi', 50),
            'macd': momentum_data.get('macd', 0),
            'current_price': current_price,
            'momentum_confirmation': momentum_confirmation
        }