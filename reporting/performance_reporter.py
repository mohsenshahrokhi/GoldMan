"""
سیستم گزارش‌دهی عملکرد
"""

from typing import List, Dict
from datetime import datetime, timedelta

try:
    import numpy as np
except ImportError:
    np = None

from database.manager import DatabaseManager


class PerformanceReporter:
    """سیستم گزارش‌دهی عملکرد"""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
    
    def calculate_win_rate(self, orders: List) -> float:
        """محاسبه Win Rate"""
        if not orders:
            return 0.0
        winning = sum(1 for t in orders if t.get('profit', 0) > 0)
        return winning / len(orders) * 100
    
    def calculate_profit_factor(self, orders: List) -> float:
        """محاسبه Profit Factor"""
        if not orders:
            return 0.0
        gross_profit = sum(t.get('profit', 0) for t in orders if t.get('profit', 0) > 0)
        gross_loss = abs(sum(t.get('profit', 0) for t in orders if t.get('profit', 0) < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        return gross_profit / gross_loss
    
    def calculate_max_drawdown(self, orders: List) -> float:
        """محاسبه Max Drawdown"""
        if not orders:
            return 0.0
        equity_curve = []
        running_equity = 0.0
        for trade in orders:
            running_equity += trade.get('profit', 0)
            equity_curve.append(running_equity)
        
        if not equity_curve:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        
        return max_dd * 100
    
    def calculate_sharpe_ratio(self, orders: List, risk_free_rate: float = 0.0) -> float:
        """محاسبه Sharpe Ratio"""
        if np is None or len(orders) < 2:
            return 0.0
        returns = [t.get('profit', 0) for t in orders]
        if not returns:
            return 0.0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return 0.0
        return (mean_return - risk_free_rate) / std_return * np.sqrt(252)  # Annualized
    
    def get_orders_report(self, symbol: str = None, strategy: str = None, 
                         start_date: datetime = None, end_date: datetime = None) -> Dict:
        """دریافت گزارش معاملات"""
        cursor = self.db.conn.cursor()
        query = "SELECT * FROM orders WHERE status = 'CLOSED'"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if strategy:
            query += " AND strategy = ?"
            params.append(strategy)
        if start_date:
            query += " AND exit_time >= ?"
            params.append(start_date)
        if end_date:
            query += " AND exit_time <= ?"
            params.append(end_date)
        
        query += " ORDER BY exit_time DESC"
        cursor.execute(query, params)
        
        orders = []
        for row in cursor.fetchall():
            orders.append({
                'ticket': row['ticket'],
                'symbol': row['symbol'],
                'direction': row['direction'],
                'entry_price': row['entry_price'],
                'exit_price': row['exit_price'],
                'profit': row['profit'],
                'entry_time': row['entry_time'],
                'exit_time': row['exit_time'],
                'strategy': row['strategy']
            })
        
        if not orders:
            return {
                'total_orders': 0,
                'winning_orders': 0,
                'losing_orders': 0,
                'total_profit': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        winning = sum(1 for t in orders if t['profit'] > 0)
        losing = len(orders) - winning
        total_profit = sum(t['profit'] for t in orders)
        
        return {
            'total_orders': len(orders),
            'winning_orders': winning,
            'losing_orders': losing,
            'total_profit': total_profit,
            'win_rate': self.calculate_win_rate(orders),
            'profit_factor': self.calculate_profit_factor(orders),
            'max_drawdown': self.calculate_max_drawdown(orders),
            'sharpe_ratio': self.calculate_sharpe_ratio(orders)
        }
    
    def generate_daily_report(self) -> str:
        """تولید گزارش روزانه"""
        today = datetime.now().date()
        start_date = datetime.combine(today, datetime.min.time())
        end_date = datetime.combine(today, datetime.max.time())
        
        report = self.get_orders_report(start_date=start_date, end_date=end_date)
        
        return f"""
📊 گزارش عملکرد روزانه - {today}

📈 آمار کلی:
• تعداد معاملات: {report['total_orders']}
• معاملات برنده: {report['winning_orders']}
• معاملات بازنده: {report['losing_orders']}
• سود/زیان کل: ${report['total_profit']:.2f}

📊 شاخص‌های عملکرد:
• Win Rate: {report['win_rate']:.2f}%
• Profit Factor: {report['profit_factor']:.2f}
• Max Drawdown: {report['max_drawdown']:.2f}%
• Sharpe Ratio: {report['sharpe_ratio']:.2f}
"""
    
    def generate_weekly_report(self) -> str:
        """تولید گزارش هفتگی"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
        
        report = self.get_orders_report(start_date=start_date, end_date=end_date)
        
        return f"""
📊 گزارش عملکرد هفتگی

📈 آمار کلی:
• تعداد معاملات: {report['total_orders']}
• معاملات برنده: {report['winning_orders']}
• معاملات بازنده: {report['losing_orders']}
• سود/زیان کل: ${report['total_profit']:.2f}

📊 شاخص‌های عملکرد:
• Win Rate: {report['win_rate']:.2f}%
• Profit Factor: {report['profit_factor']:.2f}
• Max Drawdown: {report['max_drawdown']:.2f}%
• Sharpe Ratio: {report['sharpe_ratio']:.2f}
"""
    
    def generate_monthly_report(self) -> str:
        """تولید گزارش ماهانه"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        report = self.get_orders_report(start_date=start_date, end_date=end_date)
        
        return f"""
📊 گزارش عملکرد ماهانه

📈 آمار کلی:
• تعداد معاملات: {report['total_orders']}
• معاملات برنده: {report['winning_orders']}
• معاملات بازنده: {report['losing_orders']}
• سود/زیان کل: ${report['total_profit']:.2f}

📊 شاخص‌های عملکرد:
• Win Rate: {report['win_rate']:.2f}%
• Profit Factor: {report['profit_factor']:.2f}
• Max Drawdown: {report['max_drawdown']:.2f}%
• Sharpe Ratio: {report['sharpe_ratio']:.2f}
"""

