# risk_manager.py
import numpy as np
from datetime import datetime
from config import PIP_TARGETS, SESSION_MULTIPLIERS, PAIR_VOLATILITY, EVENT_MULTIPLIERS, AI_MODEL_SETTINGS, QUICK_TRADE_PIP_TARGETS

class AdvancedRiskManager:
    @staticmethod
    def calculate_volatility_adjustment(data: dict, symbol: str, timeframe: str) -> float:
        """Fixed volatility adjustment - always returns 1.0"""
        return 1.0
    
    @staticmethod
    def calculate_atr_multiplier(data: dict, timeframe: str) -> float:
        """Fixed ATR multiplier - always returns 1.0"""
        return 1.0
    
    @staticmethod
    def calculate_pattern_multiplier(data: dict) -> float:
        """Fixed pattern multiplier - always returns 1.0"""
        return 1.0
    
    @staticmethod
    def calculate_market_condition_multiplier(data: dict) -> float:
        """Fixed market condition multiplier - always returns 1.0"""
        return 1.0
    
    @staticmethod
    def get_tactical_pip_targets(timeframe: str, symbol: str, data: dict = None) -> dict:
        """Get fixed 7 SL, 10 TP pip targets for all pairs and timeframes"""
        if timeframe not in PIP_TARGETS:
            timeframe = "1h"
        
        base_config = PIP_TARGETS[timeframe]
        
        # Fixed 7 SL, 10 TP - no adjustments
        return {
            "sl_pips": 7.0,
            "tp_pips": 10.0,
            "pip_target": 10.0,
            "hold_period": base_config["hold_period"],
            "description": base_config["description"],
            "risk_reward": base_config["risk_reward"],
            "volatility_adjustment": 1.0,
            "pattern_adjustment": 1.0,
            "combined_adjustment": 1.0,
            "base_sl": 7.0,
            "base_tp": 10.0
        }
    
    @staticmethod
    def calculate_tactical_tp_sl_levels(current_price: float, direction: int, timeframe: str, symbol: str, data: dict = None) -> dict:
        """Calculate TP/SL levels with fixed 7 SL, 10 TP targets"""
        pip_config = AdvancedRiskManager.get_tactical_pip_targets(timeframe, symbol, data)
        
        stop_loss_pips = pip_config["sl_pips"]
        take_profit_pips = pip_config["tp_pips"]
        pip_value = 0.0001
        
        if direction == 1:  # BUY
            take_profit_price = current_price + (take_profit_pips * pip_value)
            stop_loss_price = current_price - (stop_loss_pips * pip_value)
        else:  # SELL
            take_profit_price = current_price - (take_profit_pips * pip_value)
            stop_loss_price = current_price + (stop_loss_pips * pip_value)
        
        actual_ratio = take_profit_pips / stop_loss_pips if stop_loss_pips > 0 else 1.0
        
        return {
            'take_profit': round(take_profit_price, 5),
            'stop_loss': round(stop_loss_price, 5),
            'tp_pips': take_profit_pips,
            'sl_pips': stop_loss_pips,
            'reward_risk_ratio': round(actual_ratio, 2),
            'hold_period': pip_config["hold_period"],
            'pip_target': take_profit_pips,
            'description': pip_config["description"],
            'volatility_adjustment': pip_config["volatility_adjustment"],
            'pattern_adjustment': pip_config["pattern_adjustment"],
            'combined_adjustment': pip_config["combined_adjustment"],
            'risk_reward': pip_config["risk_reward"],
            'base_sl': pip_config["base_sl"],
            'base_tp': pip_config["base_tp"]
        }
    
    @staticmethod
    def calculate_tactical_position_size(account_balance: float, risk_per_trade: float, stop_loss_pips: float, confidence: str) -> tuple:
        """Calculate position size with confidence-based adjustments"""
        base_risk_amount = account_balance * (risk_per_trade / 100)
        
        # Confidence-based risk adjustment
        confidence_multiplier = {
            "high": 1.5,    # 50% more on high confidence
            "medium": 1.0,  # Standard risk
            "low": 0.6,     # 40% less on low confidence
            "very_low": 0.3,  # 70% less on very low confidence
            "quick_high": 1.2,  # 20% more for quick high confidence
            "quick_medium": 0.8,  # 20% less for quick medium
            "quick_low": 0.4  # 60% less for quick low
        }.get(confidence, 1.0)
        
        adjusted_risk = base_risk_amount * confidence_multiplier
        pip_value = 10  # Standard lot pip value
        
        # Calculate position size (in units)
        if stop_loss_pips > 0:
            position_size = adjusted_risk / (stop_loss_pips * pip_value * 0.0001)
        else:
            position_size = adjusted_risk / (7 * pip_value * 0.0001)  # Default 7 pip stop
        
        # Convert to standard units (1000 units = 0.01 lot)
        position_units = int(position_size * 1000)
        
        return position_units, confidence_multiplier

    @staticmethod
    def get_quick_trade_pip_targets(timeframe: str, symbol: str, data: dict = None) -> dict:
        """Get fixed 7 SL, 10 TP pip targets for quick trading"""
        if timeframe not in QUICK_TRADE_PIP_TARGETS:
            timeframe = "5min"
        
        base_config = QUICK_TRADE_PIP_TARGETS[timeframe]
        
        # Fixed 7 SL, 10 TP - no adjustments
        return {
            "sl_pips": 7.0,
            "tp_pips": 10.0,
            "pip_target": 10.0,
            "hold_period": base_config["hold_period"],
            "description": base_config["description"],
            "risk_reward": base_config["risk_reward"],
            "volatility_adjustment": 1.0,
            "pattern_adjustment": 1.0,
            "combined_adjustment": 1.0
        }

    @staticmethod
    def calculate_quick_trade_volatility(data: dict, timeframe: str) -> float:
        """Fixed volatility for quick trades - always returns 1.0"""
        return 1.0

class RiskManager:
    @staticmethod
    def get_pip_targets(timeframe: str) -> dict:
        """Get fixed 7 SL, 10 TP pip targets based on timeframe"""
        if timeframe not in PIP_TARGETS:
            timeframe = "1h"
        
        config = PIP_TARGETS[timeframe]
        
        return {
            "sl_pips": 7.0,  # Fixed 7 pips SL
            "tp_pips": 10.0,  # Fixed 10 pips TP
            "pip_target": 10.0,
            "hold_period": config["hold_period"],
            "description": config["description"]
        }
    
    @staticmethod
    def calculate_tp_sl_levels(current_price: float, direction: int, timeframe: str) -> dict:
        """Calculate TP/SL levels with fixed 7 SL, 10 TP values"""
        pip_config = RiskManager.get_pip_targets(timeframe)
        
        stop_loss_pips = pip_config["sl_pips"]
        take_profit_pips = pip_config["tp_pips"]
        pip_value = 0.0001
        
        if direction == 1:  # BUY
            take_profit_price = current_price + (take_profit_pips * pip_value)
            stop_loss_price = current_price - (stop_loss_pips * pip_value)
        else:  # SELL
            take_profit_price = current_price - (take_profit_pips * pip_value)
            stop_loss_price = current_price + (stop_loss_pips * pip_value)
        
        actual_ratio = take_profit_pips / stop_loss_pips if stop_loss_pips > 0 else 1.0
        
        return {
            'take_profit': round(take_profit_price, 5),
            'stop_loss': round(stop_loss_price, 5),
            'tp_pips': take_profit_pips,
            'sl_pips': stop_loss_pips,
            'reward_risk_ratio': round(actual_ratio, 2),
            'hold_period': pip_config["hold_period"],
            'pip_target': take_profit_pips,
            'description': pip_config["description"]
        }
    
    @staticmethod
    def calculate_position_size(account_balance: float, risk_per_trade: float, stop_loss_pips: int) -> int:
        """Calculate position size"""
        risk_amount = account_balance * (risk_per_trade / 100)
        pip_value = 10
        position_size = risk_amount / (stop_loss_pips * pip_value)
        return int(position_size * 1000)

def get_market_session():
    """Get current market session based on UTC time with enhanced logic"""
    hour = datetime.utcnow().hour
    minute = datetime.utcnow().minute
    
    # Enhanced session detection
    if 7 <= hour < 16: 
        return "london"
    elif 13 <= hour < 22: 
        return "new_york" 
    elif 0 <= hour < 9: 
        return "asia"
    elif (hour == 12 and minute >= 30) or (hour == 13 and minute <= 30):
        return "overlap"  # London/NY overlap
    else: 
        return "other"

def generate_trading_signals(prediction: int, patterns: dict, confidence: float) -> dict:
    """Generate enhanced trading signals based on prediction, patterns and confidence"""
    signals = []
    
    # Base signals
    if prediction == 1 and confidence > 0.6:
        signals.append("🤖 AI BUY SIGNAL")
    elif prediction == 0 and confidence > 0.6:
        signals.append("🤖 AI SELL SIGNAL")
    
    # Confidence-based recommendations
    if confidence >= 0.8:
        recommendation = "VERY HIGH CONFIDENCE - AGGRESSIVE TRADING"
        risk_level = "VERY AGGRESSIVE"
    elif confidence >= 0.7:
        recommendation = "HIGH CONFIDENCE - AGGRESSIVE TRADING"
        risk_level = "AGGRESSIVE"
    elif confidence >= 0.6:
        recommendation = "MEDIUM CONFIDENCE - MODERATE TRADING"
        risk_level = "MODERATE"
    elif confidence >= 0.5:
        recommendation = "LOW CONFIDENCE - CONSERVATIVE TRADING"
        risk_level = "CONSERVATIVE"
    else:
        recommendation = "VERY LOW CONFIDENCE - AVOID TRADING"
        risk_level = "AVOID"
    
    # Pattern-based signals
    if patterns:
        if patterns.get('double_top'):
            signals.append("⚠️ DOUBLE TOP PATTERN - Bearish Bias")
        if patterns.get('double_bottom'):
            signals.append("⚠️ DOUBLE BOTTOM PATTERN - Bullish Bias")
        if patterns.get('three_line_strike') == 1:
            signals.append("🎯 BULLISH THREE-LINE STRIKE")
        elif patterns.get('three_line_strike') == -1:
            signals.append("🎯 BEARISH THREE-LINE STRIKE")
    
    return {
        'active_signals': signals,
        'trading_recommendation': recommendation,
        'signal_strength': 'VERY STRONG' if confidence >= 0.8 else 'STRONG' if confidence >= 0.7 else 'MODERATE' if confidence >= 0.6 else 'WEAK',
        'risk_level': risk_level
    }